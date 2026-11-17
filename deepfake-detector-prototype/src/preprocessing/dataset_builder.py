"""
Walks data/extracted/<Generator>/... and builds a labels.csv with columns:
    filepath, label, generator

label is "real" if the top-level folder is "Real" (case-insensitive), else "fake".
generator is the top-level folder name (Real, DDIM, SDv15_DS0.5, etc.)

Run:
    python src/preprocessing/dataset_builder.py --root data/extracted --out data/labels.csv
"""
import argparse
import csv
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_labels(root: Path, out_csv: Path) -> int:
    rows = []
    for generator_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = generator_dir.name
        label = "real" if generator.lower() == "real" else "fake"

        # Walk recursively in case the tar unpacked into nested subfolders
        # (e.g. per-identity folders).
        for path in generator_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "filepath": str(path.resolve()),
                        "label": label,
                        "generator": generator,
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "generator"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/extracted")
    parser.add_argument("--out", type=str, default="data/labels.csv")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}. Extract your tars first.")

    n = build_labels(root, Path(args.out))
    print(f"Wrote {n} rows to {args.out}")

    # Quick sanity summary
    import pandas as pd

    df = pd.read_csv(args.out)
    print("\nPer-generator counts:")
    print(df.groupby(["generator", "label"]).size())
