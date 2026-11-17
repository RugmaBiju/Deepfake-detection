import cv2
import glob

from src.noise_analysis.high_freq import hf_residual_statistics


images = glob.glob("data/Real/*")[:5]

images += glob.glob("data/DDIM/*")[:5]


print("=" * 70)
print("TRUTHLENS HIGH-FREQUENCY TEST")
print("=" * 70)

for path in images:

    image = cv2.imread(path)

    if image is None:
        print("Could not read:", path)
        continue

    features = hf_residual_statistics(image)

    print("\n", path)

    for name, value in features.items():
        print(f"{name:25s}: {value:.8f}")


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
