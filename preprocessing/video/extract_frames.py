
import cv2
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FFPP_ROOT = Path(r"C:\FaceForensics++")

SPLIT_FILE = PROJECT_ROOT / "datasets" / "splits" / "ffpp_split.json"

OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "processed" / "ffpp_frames"

# Frame extraction settings
FRAMES_PER_VIDEO = 10
IMAGE_SIZE = 224

VIDEO_FOLDERS = {
    "real": FFPP_ROOT / "original_sequences" / "youtube" / "c23" / "videos",
    "fake": FFPP_ROOT / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
}

def extract_video_frames(video_path, output_folder):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0

    frame_indices = [
        int(x) for x in
        __import__("numpy").linspace(0, total_frames - 1, FRAMES_PER_VIDEO)
    ]

    saved = 0

    for frame_number in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        output_path = output_folder / f"frame_{frame_number:06d}.png"

        if cv2.imwrite(str(output_path), frame):
            saved += 1

    cap.release()

    return saved



with open(SPLIT_FILE, "r") as file:
    split_data = json.load(file)

records = []

for split_name in ["train", "validation", "test"]:

    print(f"\nProcessing {split_name} videos")

    groups = split_data["splits"][split_name]

    for group in tqdm(groups):

        for video_name in group["videos"]:

            label = 1 if "_" in Path(video_name).stem else 0

            category = "fake" if label == 1 else "real"

            video_path = VIDEO_FOLDERS[category] / video_name

            if not video_path.exists():
                print(f"Missing video: {video_path}")
                continue

            output_folder = (
                OUTPUT_ROOT / split_name / category /
                Path(video_name).stem
            )

            output_folder.mkdir(parents=True, exist_ok=True)

            saved_frames = extract_video_frames(
                video_path, output_folder
            )

            records.append({
                "video_name": video_name,
                "group_id": group["group_id"],
                "split": split_name,
                "label": label,
                "category": category,
                "frames_saved": saved_frames,
                "output_folder": str(output_folder)
            })

metadata = pd.DataFrame(records)

metadata_path = (
    PROJECT_ROOT / "datasets" / "metadata" /
    "ffpp_frame_metadata.csv"
)

metadata.to_csv(metadata_path, index=False)

print("\nFRAME EXTRACTION COMPLETED")
print("Videos processed:", len(metadata))
print("Frames extracted:", metadata["frames_saved"].sum())
print("Metadata saved:", metadata_path)

print("\nSplit summary:")
print(
    metadata.groupby(["split", "category"])["frames_saved"]
    .agg(["count", "sum"])
)
