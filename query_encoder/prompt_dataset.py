from __future__ import annotations
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset

class PromptClsDataset(Dataset):
    """Dataset for loading prompt-action pairs from a YAML file."""
    def __init__(self, yaml_path: str | Path):
        data = yaml.safe_load(Path(yaml_path).read_text())
        self.classes = sorted(list(data.keys()))
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (sentence, self.cls2id[action])
            for action, sentences in data.items()
            for sentence in sentences
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def class_names(self):
        """Returns the list of class names in order of their IDs."""
        return self.classes

    def num_classes(self):
        return len(self.classes)

    def __repr__(self):
        return f"PromptClsDataset({len(self.samples)} samples, {len(self.classes)} classes)"
