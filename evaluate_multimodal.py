import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_dataset import MultimodalSampleDataset, multimodal_collate
from train_multimodal import build_models


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = MultimodalSampleDataset(
        manifest_path=args.manifest,
        num_frames=args.num_frames,
        target_fps=args.target_fps,
        size=args.size,
        duration=args.duration,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate,
    )

    modules = build_models(device, args.d_model, quiet=True)
    classifier = nn.Linear(args.d_model, args.num_classes).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    modules["fusion"].load_state_dict(ckpt["fusion"])
    modules["video_proj"].load_state_dict(ckpt["video_proj"])
    modules["audio_proj"].load_state_dict(ckpt["audio_proj"])
    modules["prompt_proj"].load_state_dict(ckpt["prompt_proj"])
    classifier.load_state_dict(ckpt["classifier"])

    modules["fusion"].eval()
    classifier.eval()

    total = 0
    correct = 0
    all_loss = 0.0
    ce_loss = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Evaluating"):
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)
        audio_waveforms = [a.to(device) for a in batch["audio"]]
        transcripts = batch["transcript"]

        video_tokens = modules["video_encoder"](videos)["tokens"] if not args.drop_video else None
        if video_tokens is not None:
            video_tokens = modules["video_proj"](video_tokens)

        if args.drop_audio:
            audio_tokens = None
        else:
            audio_latent = modules["latent_encoder"](audio_waveforms, device=device, quiet=True)
            audio_tokens = modules["audio_proj"](audio_latent).unsqueeze(1)

        if args.drop_prompt:
            prompt_tokens = None
        else:
            missing_transcript = any(t is None for t in transcripts)
            if missing_transcript:
                _, transcripts = modules["speech_encoder"](audio_waveforms, device=device, quiet=True)
            prompt_tokens = modules["prompt_proj"](
                modules["prompt_encoder"](transcripts, device=device)
            ).unsqueeze(1)

        fused = modules["fusion"](
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            prompt_tokens=prompt_tokens,
        )

        logits = classifier(fused["pooled"])
        loss = ce_loss(logits, labels)

        preds = logits.argmax(dim=-1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_loss += loss.item() * labels.size(0)

    return {
        "loss": all_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multimodal fusion model")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--drop_video", action="store_true")
    parser.add_argument("--drop_audio", action="store_true")
    parser.add_argument("--drop_prompt", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = evaluate(args)
    print(f"Eval metrics: {metrics}")
