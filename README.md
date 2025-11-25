# CS7643 Project 1

This repository contains a multimodal baseline for SoccerNet clips with video, raw audio, and text prompts. It includes encoders for each modality, a transformer-based fusion module, and training/evaluation utilities.

## Environment setup
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment:
   ```bash
   conda env create -f environment.yaml
   conda activate dl-project
   ```
3. Install any missing system deps for video decoding (e.g., `ffmpeg`, `libavformat-dev`, `libavcodec-dev`, `libavdevice-dev`).

## Preparing the SoccerNet data
1. Download the SoccerNet videos and annotations from [SoccerNet](https://www.soccer-net.org/). The code expects broadcast `.mkv` videos.
2. Extract (or resample) the accompanying audio to mono WAV at 16 kHz so each video clip has a matching audio file.
3. Create a manifest (`.csv` or JSONL) with one row per training/eval clip. Required keys/columns:
   - `video`: absolute or repo-relative path to the `.mkv` file
   - `audio`: path to the matching audio file (mono WAV preferred)
   - `label`: integer class ID
   - `start_time` (optional): start second for the clip inside the full match video
   - `transcript` (optional): text transcript/ASR string; if omitted, the speech encoder will run ASR on the audio

Example CSV row:
```csv
video,audio,label,start_time,transcript
/data/SoccerNet/v1/match1.mkv,/data/SoccerNet/v1/match1.wav,2,123.5,corner kick for home side
```

## Training
Run `train_multimodal.py` with your manifest:
```bash
python train_multimodal.py \
  --manifest /data/manifests/soccernet_train.csv \
  --num_classes 17 \
  --batch_size 2 \
  --epochs 10 \
  --device cuda \
  --checkpoint_dir checkpoints
```
The script samples `num_frames` per clip via `load_video_as_tensor`, resamples audio to 16 kHz, encodes each modality, fuses the tokens, and trains with cross-entropy plus an optional contrastive term (`--contrastive_weight`). Checkpoints are saved after each epoch.

## Evaluation
Evaluate a checkpoint and optionally ablate modalities:
```bash
python evaluate_multimodal.py \
  --manifest /data/manifests/soccernet_val.csv \
  --checkpoint checkpoints/multimodal_epoch10.pt \
  --num_classes 17 \
  --device cuda
```

For ablations over video/audio/prompt presence, run:
```bash
python ablation_multimodal.py \
  --manifest /data/manifests/soccernet_val.csv \
  --checkpoint checkpoints/multimodal_epoch10.pt \
  --num_classes 17 \
  --device cuda
```

## Notes
- Clip sampling uses `video_encoder/soccernet_dataset.py::load_video_as_tensor`, which normalizes frames for VideoMAE and falls back to padding if fewer than `num_frames` are decoded.
- If transcripts are missing, `SpeechTextEncoder` will generate them from audio. Provide transcripts in the manifest to skip ASR for faster runs.
