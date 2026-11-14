
import json
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FFPP_ROOT = Path(r"C:\FaceForensics++")

SPLIT_FILE = PROJECT_ROOT / "datasets" / "splits" / "ffpp_split.json"

OUTPUT_ROOT = (
    PROJECT_ROOT / "datasets" / "processed" /
    "cross_manipulation_frames"
)

METADATA_ROOT = PROJECT_ROOT / "datasets" / "metadata"

FRAMES_PER_VIDEO = 10
IMAGE_SIZE = 224

CATEGORIES = ["Face2Face", "FaceSwap", "NeuralTextures"]

ORIGINAL_FOLDER = (
    FFPP_ROOT / "original_sequences" / "youtube" / "c23" / "videos"
)


def extract_video_frames(video_path, output_folder):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0

    frame_indices = np.linspace(
        0, total_frames - 1, FRAMES_PER_VIDEO
    ).astype(int)

    saved = 0

    for frame_number in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(
            frame, (IMAGE_SIZE, IMAGE_SIZE)
        )

        output_path = (
            output_folder / f"frame_{frame_number:06d}.png"
        )

        if cv2.imwrite(str(output_path), frame):
            saved += 1

    cap.release()

    return saved


with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)


test_groups = split_data["splits"]["test"]

records = {category: [] for category in CATEGORIES}

print("Test groups:", len(test_groups))


for group in tqdm(test_groups):

    group_id = group["group_id"]

    original_videos = [
        video for video in group["videos"]
        if "_" not in Path(video).stem
    ]

    fake_videos = [
        video for video in group["videos"]
        if "_" in Path(video).stem
    ]

    for video_name in original_videos:

        video_path = ORIGINAL_FOLDER / video_name

        output_folder = (
            OUTPUT_ROOT / "real" /
            Path(video_name).stem
        )

        output_folder.mkdir(parents=True, exist_ok=True)

        saved_frames = extract_video_frames(
            video_path, output_folder
        )

        if saved_frames != FRAMES_PER_VIDEO:
            raise RuntimeError(
                f"Frame extraction failed: {video_name}, "
                f"frames={saved_frames}"
            )

        for category in CATEGORIES:

            records[category].append({
                "video_name": video_name,
                "group_id": group_id,
                "split": "test",
                "label": 0,
                "category": "real",
                "frames_saved": saved_frames,
                "output_folder": str(output_folder)
            })


    for category in CATEGORIES:

        fake_folder = (
            FFPP_ROOT / "manipulated_sequences" /
            category / "c23" / "videos"
        )

        for video_name in fake_videos:

            video_path = fake_folder / video_name

            if not video_path.exists():
                raise FileNotFoundError(video_path)

            output_folder = (
                OUTPUT_ROOT / category /
                Path(video_name).stem
            )

            output_folder.mkdir(parents=True, exist_ok=True)

            saved_frames = extract_video_frames(
                video_path, output_folder
            )

            if saved_frames != FRAMES_PER_VIDEO:
                raise RuntimeError(
                    f"Frame extraction failed: {category}, "
                    f"{video_name}, frames={saved_frames}"
                )

            records[category].append({
                "video_name": video_name,
                "group_id": group_id,
                "split": "test",
                "label": 1,
                "category": category,
                "frames_saved": saved_frames,
                "output_folder": str(output_folder)
            })


print("\nSAVING CROSS-MANIPULATION METADATA")

for category in CATEGORIES:

    metadata = pd.DataFrame(records[category])

    output_file = (
        METADATA_ROOT /
        f"cross_manipulation_{category.lower()}_metadata.csv"
    )

    metadata.to_csv(output_file, index=False)

    print(f"\n{category}")
    print("Videos:", len(metadata))
    print("Real:", (metadata["label"] == 0).sum())
    print("Fake:", (metadata["label"] == 1).sum())
    print("Frames:", metadata["frames_saved"].sum())
    print("Metadata:", output_file)


print("\nCROSS-MANIPULATION FRAME EXTRACTION COMPLETED")
