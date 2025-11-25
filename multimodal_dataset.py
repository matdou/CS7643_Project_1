import json
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from video_encoder.soccernet_dataset import load_video_as_tensor


class MultimodalSampleDataset(Dataset):
    """Dataset that yields synchronized video/audio/text samples."""

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 16,
        target_fps: int = 16,
        size: int = 224,
        duration: float = 1.0,
        audio_sampling_rate: int = 16000,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = self._load_manifest(self.manifest_path)
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.size = size
        self.duration = duration
        self.audio_sampling_rate = audio_sampling_rate

    def _load_manifest(self, path: Path) -> list[dict]:
        if path.suffix == ".csv":
            return pd.read_csv(path).to_dict(orient="records")
        rows = []
        with open(path, "r") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        sample = self.records[idx]
        video_path = sample["video"]
        audio_path = sample["audio"]
        transcript = sample.get("transcript")
        label = sample.get("label", -1)
        start = sample.get("start_time")

        video = load_video_as_tensor(
            video_path,
            num_frames=self.num_frames,
            target_fps=self.target_fps,
            size=self.size,
            start_time=start,
            duration=self.duration,
        )

        waveform, sr = torchaudio.load(audio_path)
        waveform = torchaudio.functional.resample(waveform, sr, self.audio_sampling_rate)
        waveform = waveform.mean(dim=0)  # mono

        return {
            "video": video,
            "audio": waveform,
            "transcript": transcript,
            "label": int(label),
        }


def multimodal_collate(batch: list[dict]) -> dict:
    videos = torch.stack([b["video"] for b in batch])
    audios = [b["audio"] for b in batch]
    transcripts = [b.get("transcript") for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"video": videos, "audio": audios, "transcript": transcripts, "label": labels}
