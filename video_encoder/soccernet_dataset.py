import av
import numpy as np
import torch
import random
import cv2

# For consistent preprocessing with pretrained VideoMAE
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_video_as_tensor(
    video_path: str,
    num_frames: int = 16,
    target_fps: int = 16,
    size: int = 224,
    start_time: float = None,
    duration: float = 1.0,
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess a SoccerNet .mkv file (720p or 224p) into a VideoMAE-ready tensor.

    Args:
        video_path: path to .mkv file (SoccerNet broadcast video)
        num_frames: number of frames to sample per clip (usually 16)
        target_fps: sampling rate to approximate (VideoMAE pretrained at 16 fps)
        size: spatial crop (224x224)
        start_time: optional start second (float). Random if None.
        duration: clip duration in seconds to cover (default 1.0)
        normalize: whether to apply ImageNet mean/std normalization

    Returns:
        torch.Tensor of shape (3, num_frames, size, size)
    """

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Total duration in seconds
    total_duration = float(video_stream.duration * video_stream.time_base)
    if start_time is None:
        start_time = random.uniform(0, max(0, total_duration - duration))

    # Compute timestamps for frame selection
    frame_indices = np.linspace(
        start_time, start_time + duration, num_frames, endpoint=False
    )

    frames = []
    for ts in frame_indices:
        container.seek(int(ts / video_stream.time_base), stream=video_stream)
        frame = next(container.decode(video=0), None)
        if frame is None:
            break
        img = frame.to_rgb().to_ndarray()   # H×W×3 RGB
        img = cv2.resize(img, (size, size))
        frames.append(img)

    container.close()

    if len(frames) < num_frames:
        # pad last frame if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])

    # shape (T, H, W, C) -> (C, T, H, W)
    clip = np.stack(frames)
    clip = clip.transpose(3, 0, 1, 2)
    clip = clip / 255.0

    if normalize:
        for c in range(3):
            clip[c] = (clip[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    return torch.tensor(clip, dtype=torch.float32)
