
import json
from collections import Counter

with open("/home/hice1/mdoutre3/CS7643_Project_1/fusion/matched_pairs.json") as f:
    pairs = json.load(f)

def extract_event_name(video_path):
    filename = video_path.split("/")[-1]          # strip folder
    filename = filename.replace(".npy", "")       # remove extension
    event = filename.split("_")[-1]               # last underscore = event
    return event.strip()

counts = Counter()

for video_path, text_path in pairs:
    evt = extract_event_name(video_path)
    counts[evt] += 1

for evt, count in counts.most_common():
    print(f"{evt:20s} : {count}")
