#!/usr/bin/env python3
import pickle
import os
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------------------------------------------
# Load transcript cache (your actual format)
# -----------------------------------------------------------
def load_transcript_cache(pkl_path: Path):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    transcripts = data["transcripts"]
    metadata = data["metadata"]
    return transcripts, metadata


# -----------------------------------------------------------
# Generate embeddings PER EVENT like you do for videos
# -----------------------------------------------------------
def main():

    # Input pickle
    pkl_path = Path("audio/transcript_cache/transcripts_window_10s_centered.pkl")
    print(f"Loading: {pkl_path}")

    transcripts, metadata = load_transcript_cache(pkl_path)
    keys = list(transcripts.keys())
    print(f"Found {len(keys)} events")

    # Output directory — mirror video folder structure
    out_dir = Path("fusion/text_embeddings_events")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    print("Loading MPNet...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Process all events
    print("Encoding transcripts per event...")

    pbar = tqdm(keys)

    for key in pbar:
        meta = metadata[key]
        event_label = meta["label"] 
        raw_text    = transcripts[key]
        text = raw_text


        # determine save name
        match_dir = Path(meta["audio_path"]).parent  # e.g. ".../2017-05-13 - 14-30 Manchester City 2 - 1 Leicester"
        match_name = match_dir.name                  # e.g. "2017-05-13 - 14-30 Manchester City 2 - 1 Leicester"

        save_name = f"{match_name}_{key}.pt"
        save_path = out_dir / save_name

        if save_path.exists():
            continue

        # compute embedding
        emb = model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).cpu()

        # Save exactly like video files
        torch.save({
            "match_name": match_name,
            "event_key": key,
            "embedding": emb,                     # 768-dim tensor
            "label": meta["label"],
            "timestamp": meta["event_timestamp"],
            "audio_path": meta["audio_path"]
        }, save_path)

    print("\nDONE — text embeddings saved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
