
import json
import re
import numpy as np
import pandas as pd
import torch
import timm

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = ROOT / "datasets" / "metadata" / "ffpp_frame_metadata.csv"
MODEL_FILE = ROOT / "models" / "efficientnet_b4_best.pth"

PREDICTIONS_FILE = ROOT / "experiments" / "efficientnet_b4_test_predictions.csv"
METRICS_FILE = ROOT / "experiments" / "efficientnet_b4_test_metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2


print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=True
)

MODEL_NAME = checkpoint["model_name"]
IMAGE_SIZE = checkpoint["image_size"]

print("Model:", MODEL_NAME)
print("Image size:", IMAGE_SIZE)


metadata = pd.read_csv(METADATA_FILE)
test_metadata = metadata[metadata["split"] == "test"].copy()

print("\nTest videos:", len(test_metadata))
print("\nTest class distribution:")
print(test_metadata["label"].value_counts())


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class TestDataset(Dataset):

    def __init__(self, dataframe):
        self.samples = []

        for _, row in dataframe.iterrows():

            folder = Path(row["output_folder"])

            image_paths = [
                path for path in folder.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            image_paths = sorted(
                image_paths,
                key=lambda path: int(
                    re.findall(r"\d+", path.stem)[-1]
                )
            )

            for image_path in image_paths:
                self.samples.append((
                    image_path,
                    int(row["label"]),
                    row["video_name"]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label, video_name = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image = transform(image)

        return image, label, video_name


dataset = TestDataset(test_metadata)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTest frames:", len(dataset))


model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=2
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


video_predictions = {}

with torch.no_grad():

    for images, labels, video_names in tqdm(loader):

        images = images.to(DEVICE)

        if DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
        else:
            outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        probabilities = probabilities.cpu().numpy()

        for video_name, label, probability in zip(
            video_names,
            labels.numpy(),
            probabilities
        ):

            if video_name not in video_predictions:
                video_predictions[video_name] = {
                    "actual_label": int(label),
                    "probabilities": []
                }

            video_predictions[video_name]["probabilities"].append(
                float(probability)
            )


results = []

for video_name, data in video_predictions.items():

    fake_probability = np.mean(data["probabilities"])

    predicted_label = int(fake_probability >= 0.5)

    results.append({
        "video_name": video_name,
        "actual_label": data["actual_label"],
        "predicted_label": predicted_label,
        "fake_probability": fake_probability
    })


results_df = pd.DataFrame(results)

y_true = results_df["actual_label"]
y_pred = results_df["predicted_label"]
y_prob = results_df["fake_probability"]


metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_true, y_prob)
}


print("\nEFFICIENTNET-B4 TEST VIDEO METRICS")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Authentic", "Manipulated"],
    zero_division=0
))


results_df.to_csv(PREDICTIONS_FILE, index=False)

with open(METRICS_FILE, "w") as file:
    json.dump(metrics, file, indent=4)


print("\nTEST EVALUATION COMPLETED")
print("Videos evaluated:", len(results_df))
print("Predictions saved:", PREDICTIONS_FILE)
print("Metrics saved:", METRICS_FILE)
