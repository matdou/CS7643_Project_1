# query_encoder/clip_query_encoder.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import yaml
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Encoder running on device = {DEVICE}")

class CLIPQueryEncoder:
    """
    CLIP ViT-B/32 text encoder with optional few-shot adapter.
    - encode_query(texts) -> L2-normalized CLIP embeddings [B, 512]
    - get_embedding(texts, adapted=True) -> 512-D embedding for fusion (raw or adapter-projected)
    - logits(texts) -> class logits [B, C] if adapter is loaded
    - topk(text, k) -> [(class, prob)] using adapter (if loaded)
    - build_class_matrix(prompts_yaml) -> [C, 512] class prototypes (avg of paraphrases)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        adapter_path: Optional[str | Path] = None,
        classes_path: Optional[str | Path] = None,
    ):
        # CLIP text encoder (frozen) 
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval().to(DEVICE)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        for p in self.model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            d = int(self.model.text_projection.shape[1])  # 512 for ViT-B/32
        self.embed_dim = d

        # few-shot adapter (Linear 512 -> C) 
        self.adapter: Optional[nn.Linear] = None
        self.class_names: Optional[List[str]] = None
        if adapter_path is not None and classes_path is not None:
            self.load_adapter(adapter_path, classes_path)

    @torch.no_grad()
    def encode_query(self, texts: List[str]) -> torch.Tensor:
        """Raw CLIP text embedding (L2-normalized) on DEVICE: [B, 512]."""
        toks = self.tokenizer(texts).to(DEVICE)
        z = self.model.encode_text(toks).float()
        z = z / z.norm(dim=-1, keepdim=True)
        return z

    @torch.no_grad()
    def get_embedding(self, texts: List[str], adapted: bool = True) -> torch.Tensor:
        """
        Embedding to feed the cross-attention transformer.
        - If adapted=True and adapter is loaded: project through adapter weight basis (512->C, back to 512 via W^T).
        - Else: return raw CLIP 512-D.
        Always L2-normalized on output.
        """
        q = self.encode_query(texts)  # [B,512]
        if adapted and self.adapter is not None:
            # projection via adapter weight basis (keeps 512-D, domain-aligned)
            W = self.adapter.weight.T  # [512, C]
            q = torch.matmul(q, W)     # [B,512]
            q = q / q.norm(dim=-1, keepdim=True)
        return q

    def load_adapter(self, adapter_path: str | Path, classes_path: str | Path):
        """Load few-shot adapter weights (Linear 512->C) and class list."""
        classes = yaml.safe_load(Path(classes_path).read_text()) if str(classes_path).endswith((".yaml", ".yml")) \
                  else __import__("json").loads(Path(classes_path).read_text())
        C = len(classes)
        head = nn.Linear(512, C).to(DEVICE)
        state = torch.load(adapter_path, map_location=DEVICE)
        head.load_state_dict(state)
        head.eval()
        self.adapter = head
        self.class_names = list(classes)

    @torch.no_grad()
    def logits(self, texts: List[str]) -> torch.Tensor:
        """Adapter class logits [B, C]. Requires adapter loaded."""
        assert self.adapter is not None, "Adapter not loaded. Call load_adapter()."
        q = self.encode_query(texts)    # [B,512]
        return self.adapter(q)          # [B,C]

    @torch.no_grad()
    def topk(self, text: str, k: int = 3):
        """Top-k class predictions using adapter probs."""
        assert self.adapter is not None and self.class_names is not None, "Adapter/classes not loaded."
        probs = self.logits([text]).softmax(-1)[0]
        k = min(k, len(self.class_names))
        topv, topi = probs.topk(k)
        return [(self.class_names[int(i)], float(v)) for v, i in zip(topv, topi)]
