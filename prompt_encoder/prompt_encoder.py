from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class PromptEncoder(nn.Module):
    """Text encoder with a clear interface for prompt embeddings."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | torch.device | None = None,
        quiet: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.quiet = quiet
        self.to(self.device)

    @property
    def output_dim(self) -> int:
        return self.model.config.hidden_size

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def forward(self, prompts: List[str], device: str | torch.device | None = None) -> torch.Tensor:
        run_device = torch.device(device) if device is not None else self.device
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(run_device) for k, v in inputs.items()}
        self.model.to(run_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        self._log(f"[PromptEncoder] embeddings={embeddings.shape}")
        return embeddings
