###################################
# Training of the few-shot adapter#
###################################

from __future__ import annotations
import argparse, json, yaml, torch, random
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from .prompt_dataset import PromptClsDataset
from .clip_query_encoder import CLIPQueryEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[adapter] Running on device: {DEVICE}")

def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def embed_texts(encoder: CLIPQueryEncoder, texts: list[str]) -> torch.Tensor:
    """
    Encode a list of sentences with the frozen CLIP text tower.
    Returns a [B, 512] L2-normalized tensor on the current DEVICE.
    """
    with torch.no_grad():
        z = encoder.encode_query(texts)
    return z

def eval_split(encoder: CLIPQueryEncoder, head: nn.Module, yaml_path: str | Path, topk: tuple[int,int] = (1,3)) -> dict:
    """
    Evaluate the linear head on a YAML split (train/test) of prompts.
    Computes top-1 and top-3 accuracy over all sentences.
    """
    ds = PromptClsDataset(yaml_path)
    C = ds.num_classes()
    k_eval = min(3, C) 

    B = 128
    total = 0
    hit1 = 0
    hitk = 0

    head.eval()
    for i in range(0, len(ds), B):
        batch = ds.samples[i:i+B]
        texts = [t for t, _ in batch]
        y = torch.tensor([y for _, y in batch], device=DEVICE)
        q = embed_texts(encoder, texts)              # [B,512] on DEVICE
        logits = head(q)                             # [B,C] on DEVICE
        probs = logits.softmax(-1)

        topv, topi = probs.topk(k_eval, dim=-1)      # [B,k]
        hit1 += (topi[:, 0] == y).sum().item()
        hitk += (topi == y.unsqueeze(1)).any(dim=1).sum().item()
        total += len(batch)

    out = {
        "top1": hit1 / total if total else 0.0,
        f"top{k_eval}": hitk / total if total else 0.0
    }
    return out

def train_adapter(prompts_train: str | Path,
                  prompts_test: str | Path,
                  out_dir: str | Path,
                  epochs: int = 8,
                  bs: int = 64,
                  lr: float = 1e-3,
                  wd: float = 0.01,
                  seed: int = 42):
    """
    Train a few-shot linear adapter (512 -> #classes) on prompts_train.yaml and
    evaluate on prompts_test.yaml. Saves best weights and class list.
    """
    set_seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    #Frozen CLIP text encoder and datasets
    enc = CLIPQueryEncoder()                         # CLIP text tower (frozen)
    train_ds = PromptClsDataset(prompts_train)
    classes = train_ds.class_names()
    C = len(classes)

    # Embed all training texts once
    texts, labels = zip(*train_ds.samples)         
    X = embed_texts(enc, list(texts)).detach()   
    y = torch.tensor(labels, device=DEVICE)
    train_data = list(zip(X, y))
    loader = DataLoader(train_data, batch_size=bs, shuffle=True)

    #Linear head and optimizer
    head = nn.Linear(512, C).to(DEVICE)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()

    print(f"[info] classes: {C} | train samples: {len(train_ds)} | epochs: {epochs}")

    #Train loop with eval on test split
    best_top1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        head.train()
        tot, correct, running_loss = 0, 0, 0.0

        for xb, yb in loader:
            xb = xb.to(DEVICE)                       # [B,512]
            yb = yb.to(DEVICE)                       # [B]
            logits = head(xb)                        # [B,C]
            l = loss(logits, yb)

            opt.zero_grad()
            l.backward()
            opt.step()

            running_loss += float(l) * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(xb)

        tr_acc = correct / tot if tot else 0.0
        tr_loss = running_loss / tot if tot else 0.0

        # eval on the harder, realistic test prompts
        metrics = eval_split(enc, head, prompts_test, topk=(1,3))
        top1 = metrics["top1"]; topk_key = next(k for k in metrics.keys() if k.startswith("top") and k != "top1")
        topk_val = metrics[topk_key]
        print(f"[ep {ep}] train acc {tr_acc:.3f} loss {tr_loss:.3f} | test top1 {top1:.3f} {topk_key} {topk_val:.3f}")

        if top1 > best_top1:
            best_top1 = top1
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    # Save best weights and classes
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}

    torch.save(best_state, out / "adapter_linear.pt")
    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    print(f"Saved adapter to {out} | Best test top1={best_top1:.3f}")

def predict_free_text(sentence: str, adapter_path: str | Path, classes_path: str | Path):
    """
    Run the trained adapter on a single sentence and return top-3 (class, prob).
    """
    enc = CLIPQueryEncoder()
    classes = json.loads(Path(classes_path).read_text())
    head = nn.Linear(512, len(classes)).to(DEVICE)
    head.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
    head.eval()
    q = embed_texts(enc, [sentence])                 # [1,512] on DEVICE
    probs = head(q).softmax(-1)[0]                   # [C]
    k = min(3, len(classes))
    topv, topi = probs.topk(k)
    return [(classes[int(i)], float(v)) for v, i in zip(topv, topi)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a few-shot linear adapter on CLIP text embeddings.")
    ap.add_argument("--train", default='query_encoder/prompts_train.yaml', help="Path to prompts_train.yaml")
    ap.add_argument("--test",  default='query_encoder/prompts_test.yaml',  help="Path to prompts_test.yaml")
    ap.add_argument("--out",   default="query_encoder/adapter", help="Output directory")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs",     type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-2)
    ap.add_argument("--wd",     type=float, default=0.01)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()
    train_adapter(args.train, args.test, args.out, args.epochs, args.bs, args.lr, args.wd, args.seed)