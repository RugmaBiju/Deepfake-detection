
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
    "cross_manipulation_frames" / "train_validation"
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

        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        output_path = (
            output_folder / f"frame_{frame_number:06d}.png"
        )

        if cv2.imwrite(str(output_path), frame):
            saved += 1

    cap.release()

    return saved


def get_fake_videos(group):

    return [
        video for video in group["videos"]
        if "_" in Path(video).stem
    ]


def get_original_videos(group):

    return [
        video for video in group["videos"]
        if "_" not in Path(video).stem
    ]


def is_group_complete(group):

    fake_videos = get_fake_videos(group)

    for category in CATEGORIES:

        fake_folder = (
            FFPP_ROOT / "manipulated_sequences" /
            category / "c23" / "videos"
        )

        for video_name in fake_videos:

            if not (fake_folder / video_name).exists():
                return False

    return True


with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)


records = {category: [] for category in CATEGORIES}

METADATA_ROOT.mkdir(parents=True, exist_ok=True)

for split_name in ["train", "validation"]:

    groups = split_data["splits"][split_name]

    print("\n" + "=" * 55)
    print("CHECKING:", split_name.upper())
    print("Original groups:", len(groups))
    print("=" * 55)

    complete_groups = []
    excluded_groups = []

    for group in groups:

        if is_group_complete(group):
            complete_groups.append(group)
        else:
            excluded_groups.append(group)

    if split_name == "validation" and excluded_groups:

        for group in excluded_groups:
            print("Incomplete validation group:", group["group_id"])

        raise RuntimeError(
            "Validation contains missing manipulation videos. "
            "Extraction stopped."
        )

    print("Complete groups:", len(complete_groups))
    print("Excluded groups:", len(excluded_groups))

    for group in excluded_groups:

        missing_files = []

        for category in CATEGORIES:

            fake_folder = (
                FFPP_ROOT / "manipulated_sequences" /
                category / "c23" / "videos"
            )

            for video_name in get_fake_videos(group):

                if not (fake_folder / video_name).exists():
                    missing_files.append(
                        f"{category}/{video_name}"
                    )

        print("\nExcluded group:", group["group_id"])
        print("Missing files:", missing_files)

    print("\n" + "=" * 55)
    print("PROCESSING:", split_name.upper())
    print("Groups:", len(complete_groups))
    print("=" * 55)

    for group in tqdm(complete_groups):

        group_id = group["group_id"]

        original_videos = get_original_videos(group)
        fake_videos = get_fake_videos(group)

        for video_name in original_videos:

            video_path = ORIGINAL_FOLDER / video_name

            if not video_path.exists():
                raise FileNotFoundError(video_path)

            output_folder = (
                OUTPUT_ROOT / split_name / "real" /
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
                    "split": split_name,
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
                    OUTPUT_ROOT / split_name / category /
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
                    "split": split_name,
                    "label": 1,
                    "category": category,
                    "frames_saved": saved_frames,
                    "output_folder": str(output_folder)
                })


print("\nSAVING TRAINING AND VALIDATION METADATA")

for category in CATEGORIES:

    metadata = pd.DataFrame(records[category])

    output_file = (
        METADATA_ROOT /
        f"cross_manipulation_{category.lower()}_trainval_metadata.csv"
    )

    metadata.to_csv(output_file, index=False)

    print("\n", category)
    print("Total videos:", len(metadata))
    print(
        "Training videos:",
        (metadata["split"] == "train").sum()
    )
    print(
        "Validation videos:",
        (metadata["split"] == "validation").sum()
    )
    print("Real videos:", (metadata["label"] == 0).sum())
    print("Fake videos:", (metadata["label"] == 1).sum())
    print("Total frames:", metadata["frames_saved"].sum())
    print("Metadata:", output_file)


print("\nTRAINING AND VALIDATION FRAME EXTRACTION COMPLETED")
