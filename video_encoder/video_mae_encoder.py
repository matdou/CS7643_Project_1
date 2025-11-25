import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, VideoMAEModel


class VideoMAEEncoder(nn.Module):
    """VideoMAE wrapper that outputs per-temporal-chunk tokens."""

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__()
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.quiet = quiet
        self.to(self.device)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def forward(self, video: torch.Tensor, return_sequence: bool = True) -> dict:
        """
        Args:
            video: Tensor of shape [B, 3, T, H, W] in RGB, normalized.
        Returns:
            dict with ``tokens`` [B, Tv, D] and optional ``sequence`` [B, S, D].
        """

        b, c, t, h, w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).cpu().numpy()
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        self.model.to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values)

        sequence = outputs.last_hidden_state  # [B, 1 + tokens, D]
        cls_token, patch_tokens = sequence[:, :1], sequence[:, 1:]

        patch_size = self.model.config.patch_size if isinstance(self.model.config.patch_size, int) else self.model.config.patch_size[0]
        tubelet = self.model.config.tubelet_size if isinstance(self.model.config.tubelet_size, (list, tuple)) else (self.model.config.tubelet_size,)
        spatial_tokens = (self.model.config.image_size // patch_size) ** 2

        temporal_tokens = patch_tokens.shape[1] // spatial_tokens
        patch_tokens = patch_tokens.view(b, temporal_tokens, spatial_tokens, -1)
        tokens = patch_tokens.mean(dim=2)  # [B, temporal_tokens, D]
        self._log(f"[VideoMAE] tokens={tokens.shape}, cls={cls_token.shape}")

        return {"tokens": tokens, "cls": cls_token, "sequence": sequence} if return_sequence else {"tokens": tokens}
