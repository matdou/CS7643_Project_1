import subprocess
import os

# Root folders
input_root = "../SoccerNetData"
output_root = "../SoccerNetDataAudio"

# Loop through every file recursively
for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(".mkv"):
            video_path = os.path.join(root, file)

            # Get parent folder (the match folder, e.g., "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley")
            parent_folder = os.path.basename(os.path.dirname(video_path))

            # Create output folder for this match
            output_dir = os.path.join(output_root, parent_folder)
            os.makedirs(output_dir, exist_ok=True)

            # Build output file path
            base_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"{base_name}.wav")

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"Audio already exists: {output_path}")
                continue

            # Build ffmpeg command
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_path
            ]

            # Run command safely
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ Extracted: {output_path}")
            except subprocess.CalledProcessError as e:
                print(f" Failed on {video_path}: {e}")
