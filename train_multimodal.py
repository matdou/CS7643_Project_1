import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio.audio_encoder.combined_audio_encoder import MultiModalFusionTransformer
from audio.audio_encoder.latent_audio_encoder import LatentAudioEncoder
from audio.audio_encoder.speech_text_encoder import SpeechTextEncoder
from multimodal_dataset import MultimodalSampleDataset, multimodal_collate
from prompt_encoder import PromptEncoder
from video_encoder.video_mae_encoder import VideoMAEEncoder


def build_models(device: torch.device, d_model: int, quiet: bool = True):
    video_encoder = VideoMAEEncoder(device=device, quiet=quiet)
    latent_encoder = LatentAudioEncoder(device=device, quiet=quiet)
    speech_encoder = SpeechTextEncoder(device=device, quiet=quiet)
    prompt_encoder = PromptEncoder(device=device, quiet=quiet)

    fusion = MultiModalFusionTransformer(d_model=d_model)

    video_proj = nn.Linear(video_encoder.hidden_size, d_model)
    audio_proj = nn.Linear(latent_encoder.output_dim, d_model)
    prompt_proj = nn.Linear(prompt_encoder.output_dim, d_model)

    return {
        "video_encoder": video_encoder,
        "latent_encoder": latent_encoder,
        "speech_encoder": speech_encoder,
        "prompt_encoder": prompt_encoder,
        "fusion": fusion,
        "video_proj": video_proj,
        "audio_proj": audio_proj,
        "prompt_proj": prompt_proj,
    }


def contrastive_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    anchor = nn.functional.normalize(anchor, dim=-1)
    positive = nn.functional.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return nn.functional.cross_entropy(logits, labels)


def train(args: argparse.Namespace) -> None:
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
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate,
    )

    modules = build_models(device, args.d_model, quiet=args.quiet)
    classifier = nn.Linear(args.d_model, args.num_classes).to(device)

    params = (
        list(modules["fusion"].parameters())
        + list(modules["video_proj"].parameters())
        + list(modules["audio_proj"].parameters())
        + list(modules["prompt_proj"].parameters())
        + list(classifier.parameters())
    )
    optimizer = optim.Adam(params, lr=args.lr)

    ce_loss = nn.CrossEntropyLoss()

    modules["fusion"].train()
    classifier.train()

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_acc = 0.0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            videos = batch["video"].to(device)
            labels = batch["label"].to(device)
            audio_waveforms = [a.to(device) for a in batch["audio"]]

            # Video tokens
            with torch.no_grad():
                video_tokens = modules["video_encoder"](videos)["tokens"]
            video_tokens = modules["video_proj"](video_tokens)

            # Audio latent tokens
            audio_latent = modules["latent_encoder"](audio_waveforms, device=device, quiet=True)
            audio_tokens = modules["audio_proj"](audio_latent).unsqueeze(1)

            # Transcripts or ASR
            transcripts = batch["transcript"]
            missing_transcript = any(t is None for t in transcripts)
            if missing_transcript:
                speech_emb, transcripts = modules["speech_encoder"](
                    audio_waveforms, device=device, quiet=True
                )
            else:
                speech_emb = modules["speech_encoder"].encode_text(transcripts, device=device, quiet=True)

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

            if args.contrastive_weight > 0:
                loss = loss + args.contrastive_weight * contrastive_loss(
                    fused["pooled"], prompt_tokens.squeeze(1)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=-1)
            running_acc += (preds == labels).float().mean().item()

            if (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                avg_acc = running_acc / args.log_interval
                print(f"Step {step+1}: loss={avg_loss:.4f} acc={avg_acc:.4f}")
                running_loss = 0.0
                running_acc = 0.0

        save_path = Path(args.checkpoint_dir) / f"multimodal_epoch{epoch+1}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fusion": modules["fusion"].state_dict(),
                "classifier": classifier.state_dict(),
                "video_proj": modules["video_proj"].state_dict(),
                "audio_proj": modules["audio_proj"].state_dict(),
                "prompt_proj": modules["prompt_proj"].state_dict(),
            },
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal fusion model")
    parser.add_argument("--manifest", type=str, required=True, help="JSONL/CSV manifest")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--target_fps", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
