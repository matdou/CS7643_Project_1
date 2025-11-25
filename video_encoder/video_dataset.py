import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .soccernet_dataset import load_video_as_tensor


class VideoClipDataset(Dataset):
    """Load video clips plus labels using ``load_video_as_tensor``."""

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 16,
        target_fps: int = 16,
        size: int = 224,
        duration: float = 1.0,
        transform: Optional[Callable] = None,
        label_map: Optional[dict] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.samples = self._load_manifest(self.manifest_path)
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.size = size
        self.duration = duration
        self.transform = transform
        self.label_map = label_map

    def _load_manifest(self, path: Path) -> list[dict]:
        if path.suffix in {".csv"}:
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        elif path.suffix in {".json", ".jsonl"}:
            records = []
            with open(path, "r") as f:
                for line in f:
                    records.append(json.loads(line))
            return records
        else:
            raise ValueError(f"Unsupported manifest format: {path.suffix}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        video_path = sample["video"]
        label = sample.get("label", -1)
        start = sample.get("start_time")

        clip = load_video_as_tensor(
            video_path,
            num_frames=self.num_frames,
            target_fps=self.target_fps,
            size=self.size,
            start_time=start,
            duration=self.duration,
        )

        if self.transform:
            clip = self.transform(clip)

        if self.label_map is not None and label in self.label_map:
            label = self.label_map[label]

        return {"video": clip, "label": torch.tensor(label, dtype=torch.long)}
