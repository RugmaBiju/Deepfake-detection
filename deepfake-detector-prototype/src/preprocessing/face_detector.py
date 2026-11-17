"""
Face detection + alignment using OpenCV's bundled Haar cascade.

Why Haar cascade instead of MediaPipe/RetinaFace for this prototype:
MediaPipe's classic `solutions.face_detection` API has been deprecated
and recent pip builds (0.10.x) may not ship it at all, depending on
platform/Python version — it silently breaks with an AttributeError.
The newer MediaPipe Tasks API works but requires downloading a separate
.tflite model file at runtime. Haar cascade ships inside opencv-python
itself (no extra downloads, no version roulette), so it's the more
reliable choice to get a prototype running today. It's less accurate on
extreme poses/lighting than RetinaFace, which is worth swapping in later
once the pipeline is validated end-to-end.

Usage:
    from src.preprocessing.face_detector import FaceCropper
    cropper = FaceCropper()
    crop = cropper.crop(image_bgr)   # returns aligned face crop or None
"""
import cv2
import numpy as np


class FaceCropper:
    def __init__(self, output_size: int = 256, margin: float = 0.35):
        """
        output_size: side length (px) of the square output crop.
        margin: extra context added around the tight face bounding box,
                as a fraction of the box size. PRNU/noise stats benefit
                from a bit of context beyond just eyes-nose-mouth.
        """
        self.output_size = output_size
        self.margin = margin
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def crop(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        # Take the largest detected face box (most likely the main subject)
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])

        # Add margin
        mx = bw * self.margin
        my = bh * self.margin
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if x2 <= x1 or y2 <= y1:
            return None

        face = image_bgr[y1:y2, x1:x2]
        face = cv2.resize(face, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)
        return face

    def close(self):
        pass  # kept for API compatibility with callers that call cropper.close()


if __name__ == "__main__":
    # Quick manual test: point this at any local image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_detector.py <path_to_image>")
        raise SystemExit(1)

    img = cv2.imread(sys.argv[1])
    cropper = FaceCropper()
    face = cropper.crop(img)
    if face is None:
        print("No face detected.")
    else:
        out_path = "face_crop_debug.png"
        cv2.imwrite(out_path, face)
        print(f"Saved crop to {out_path}, shape={face.shape}")
