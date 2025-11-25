import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class LatentAudioEncoder(nn.Module):
    """Wrapper around Wav2Vec2 that supports batching and device control."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-960h-lv60-self",
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name, torch_dtype=torch.float32)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.quiet = quiet
        self.to(self.device)

    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size

    def _log(self, msg: str, quiet: bool) -> None:
        if not (self.quiet or quiet):
            print(msg)

    def _prepare_waveforms(self, waveform: torch.Tensor | np.ndarray | list) -> list:
        # Normalize inputs into a list of 1D numpy arrays
        if isinstance(waveform, torch.Tensor):
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            waveform_list = [w.detach().cpu().numpy() for w in waveform]
        elif isinstance(waveform, np.ndarray):
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            waveform_list = [w for w in waveform]
        elif isinstance(waveform, list):
            waveform_list = []
            for w in waveform:
                if isinstance(w, torch.Tensor):
                    waveform_list.append(w.detach().cpu().numpy())
                else:
                    waveform_list.append(np.asarray(w))
        else:
            raise TypeError(f"Unsupported waveform type: {type(waveform)}")

        processed = []
        for w in waveform_list:
            if w.size == 0:
                w = np.zeros(320, dtype=np.float32)
            if w.ndim > 1:
                w = np.squeeze(w)
            if w.shape[0] < 320:
                pad_len = 320 - w.shape[0]
                w = np.pad(w, (0, pad_len))
            processed.append(w.astype(np.float32))

        return processed

    def forward(
        self,
        waveform: torch.Tensor | np.ndarray | list,
        sampling_rate: int = 16000,
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> torch.Tensor:
        """Encode waveform(s) to latent representations.

        Args:
            waveform: Tensor/array/list with shape ``[B, T]`` or ``[T]``.
            sampling_rate: Input sampling rate.
            device: Optional device override for computation.
            quiet: If True, suppress verbose logging.

        Returns:
            torch.Tensor: shape ``[B, hidden]`` pooled embeddings.
        """

        run_device = torch.device(device) if device is not None else self.device
        self._log(f"[LATENT] Using device={run_device}", quiet)

        waveforms = self._prepare_waveforms(waveform)
        inputs = self.processor(
            waveforms,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(run_device) for k, v in inputs.items()}
        self.model.to(run_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)
        self._log(f"[LATENT] Output embedding shape: {embedding.shape}", quiet)
        return embedding