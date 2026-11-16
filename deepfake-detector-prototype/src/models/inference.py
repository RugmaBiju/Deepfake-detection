"""
Loads the trained baseline model and predicts on new image(s).

Run (single image):
    python src/models/inference.py --model models_saved/baseline_xgb.pkl --image path/to/face.jpg

Run (folder):
    python src/models/inference.py --model models_saved/baseline_xgb.pkl --folder path/to/images/
"""
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.face_detector import FaceCropper
from src.features.feature_builder import build_features_for_image


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["feature_columns"]


def predict_image(image_path: str, model, feature_columns, cropper: FaceCropper) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {"filepath": image_path, "error": "could not read image"}

    face = cropper.crop(img)
    if face is None:
        return {"filepath": image_path, "error": "no face detected"}

    feats = build_features_for_image(face)
    x = np.array([[feats[c] for c in feature_columns]])

    prob_fake = float(model.predict_proba(x)[0, 1])
    verdict = "FAKE" if prob_fake >= 0.5 else "REAL"

    # Simple per-branch explanation: which feature group deviates most
    # from typical real-image ranges is left as a placeholder — swap in
    # actual real-image reference stats from your training set for a
    # more meaningful explanation in the demo UI.
    return {
        "filepath": image_path,
        "verdict": verdict,
        "confidence": prob_fake if verdict == "FAKE" else 1 - prob_fake,
        "prob_fake": prob_fake,
        "features": feats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--folder", type=str, help="Folder of images")
    args = parser.parse_args()

    model, feature_columns = load_model(args.model)
    cropper = FaceCropper(output_size=256)

    targets = []
    if args.image:
        targets = [args.image]
    elif args.folder:
        targets = [str(p) for p in Path(args.folder).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    else:
        raise SystemExit("Provide --image or --folder")

    for t in targets:
        result = predict_image(t, model, feature_columns, cropper)
        if "error" in result:
            print(f"{t}: SKIPPED ({result['error']})")
        else:
            print(f"{t}: {result['verdict']} (prob_fake={result['prob_fake']:.3f})")

    cropper.close()
