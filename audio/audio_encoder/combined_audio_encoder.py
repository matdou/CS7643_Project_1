import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusionTransformer(nn.Module):
    """Transformer-based fusion for video/audio/prompt tokens with masking support."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.video_modality = nn.Parameter(torch.randn(1, 1, d_model))
        self.audio_modality = nn.Parameter(torch.randn(1, 1, d_model))
        self.prompt_modality = nn.Parameter(torch.randn(1, 1, d_model))

    def _add_modality(self, tokens: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return tokens + embedding

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
        prompt_tokens: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> dict:
        """
        Args:
            video_tokens: Tensor [B, Tv, D]
            audio_tokens: Tensor [B, Ta, D]
            prompt_tokens: Tensor [B, Tp, D]
            *_mask: Bool Tensor [B, T] with True for padding positions
        Returns:
            dict with ``sequence`` [B, T, D] and ``pooled`` [B, D]
        """

        token_chunks = []
        padding_masks = []

        if video_tokens is not None:
            token_chunks.append(self._add_modality(video_tokens, self.video_modality))
            padding_masks.append(video_mask if video_mask is not None else torch.zeros(video_tokens.shape[:2], dtype=torch.bool, device=video_tokens.device))

        if audio_tokens is not None:
            token_chunks.append(self._add_modality(audio_tokens, self.audio_modality))
            padding_masks.append(audio_mask if audio_mask is not None else torch.zeros(audio_tokens.shape[:2], dtype=torch.bool, device=audio_tokens.device))

        if prompt_tokens is not None:
            token_chunks.append(self._add_modality(prompt_tokens, self.prompt_modality))
            padding_masks.append(prompt_mask if prompt_mask is not None else torch.zeros(prompt_tokens.shape[:2], dtype=torch.bool, device=prompt_tokens.device))

        if not token_chunks:
            raise ValueError("At least one modality token tensor must be provided")

        tokens = torch.cat(token_chunks, dim=1)  # [B, T, D]
        mask = torch.cat(padding_masks, dim=1) if padding_masks else None

        encoded = self.encoder(tokens.transpose(0, 1), src_key_padding_mask=mask)
        encoded = encoded.transpose(0, 1)

        if mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = (~mask).float().sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (encoded * (~mask).unsqueeze(-1)).sum(dim=1) / valid

        outputs = {"pooled": pooled}
        if return_sequence:
            outputs["sequence"] = encoded
        return outputs
