import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class SpeechTextEncoder(nn.Module):
    """Speech-to-text transcription + text embedding with batching support."""

    def __init__(
        self,
        asr_model: str = "openai/whisper-base",
        text_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        translate_to_english: bool = True,
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__()
        self.translate_to_english = translate_to_english
        self.whisper_processor = WhisperProcessor.from_pretrained(asr_model)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(asr_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.quiet = quiet
        self.to(self.device)

    @property
    def output_dim(self) -> int:
        return self.text_model.config.hidden_size

    def _log(self, msg: str, quiet: bool) -> None:
        if not (self.quiet or quiet):
            print(msg)

    def _normalize_waveforms(self, waveform: torch.Tensor | np.ndarray | list) -> list:
        if isinstance(waveform, torch.Tensor):
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            data = [w.detach().cpu().numpy() for w in waveform]
        elif isinstance(waveform, np.ndarray):
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            data = [w for w in waveform]
        elif isinstance(waveform, list):
            data = []
            for w in waveform:
                if isinstance(w, torch.Tensor):
                    data.append(w.detach().cpu().numpy())
                else:
                    data.append(np.asarray(w))
        else:
            raise TypeError(f"Unsupported waveform type: {type(waveform)}")

        processed = []
        for w in data:
            if w.ndim > 1:
                w = np.squeeze(w)
            processed.append(w.astype(np.float32))
        return processed

    def transcribe(
        self,
        waveform: torch.Tensor | np.ndarray | list,
        sampling_rate: int = 16000,
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> list[str]:
        run_device = torch.device(device) if device is not None else self.device
        waveforms = self._normalize_waveforms(waveform)
        features = self.whisper_processor(
            waveforms, sampling_rate=sampling_rate, return_tensors="pt", padding=True
        )
        features = {k: v.to(run_device) for k, v in features.items()}
        self.whisper.to(run_device)

        generate_opts = {"task": "translate"} if self.translate_to_english else {}
        with torch.no_grad():
            ids = self.whisper.generate(features["input_features"], **generate_opts)

        transcripts = [t.strip() for t in self.whisper_processor.batch_decode(ids, skip_special_tokens=True)]
        self._log(f"[SPEECH] Transcribed {len(transcripts)} utterances", quiet)
        return transcripts

    def encode_text(
        self,
        texts: list[str],
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> torch.Tensor:
        run_device = torch.device(device) if device is not None else self.device
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(run_device) for k, v in inputs.items()}
        self.text_model.to(run_device)

        with torch.no_grad():
            outputs = self.text_model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        self._log(f"[SPEECH] Text embedding shape: {embeddings.shape}", quiet)
        return embeddings

    def forward(
        self,
        waveform: torch.Tensor | np.ndarray | list,
        sampling_rate: int = 16000,
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> tuple[torch.Tensor, list[str]]:
        """Transcribe speech and return pooled text embeddings.

        Returns:
            embeddings: Tensor of shape ``[B, hidden]``
            transcripts: list of transcribed strings
        """

        transcripts = self.transcribe(
            waveform, sampling_rate=sampling_rate, device=device, quiet=quiet
        )
        embeddings = self.encode_text(transcripts, device=device, quiet=quiet)
        return embeddings, transcripts