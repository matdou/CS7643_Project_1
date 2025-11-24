import torch
import torch.nn as nn

class CombinedAudioEncoder(nn.Module):
    def __init__(self, audio_dim=1024, text_dim=384, fused_dim=512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.fusion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim)
        )

    def forward(self, audio_emb, text_emb):
        a = self.audio_proj(audio_emb)
        t = self.text_proj(text_emb)
        combined = torch.cat((a, t), dim=-1)
        fused = self.fusion(combined)
        return fused
