import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "prnu_block_mean",
    "prnu_block_std",
    "prnu_block_min",
    "pg_a",
    "pg_b",
    "pg_fit_rmse",
    "pg_num_flat_patches",
    "hf_energy_ratio",
    "hf_peak_concentration",
    "hf_peak_max",
]

df = pd.read_csv("data/features.csv")

print("\n" + "=" * 70)
print("TRUTHLENS FEATURE ANALYSIS")
print("=" * 70)

print(f"\nTotal samples: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nGenerator distribution:")
print(df["generator"].value_counts())

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

for col in FEATURE_COLUMNS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for feature in FEATURE_COLUMNS:
    print(f"\n--- {feature} ---")

    real = df[df["label"] == "real"][feature].dropna()
    fake = df[df["label"] == "fake"][feature].dropna()

    print(f"REAL : mean={real.mean():.6f}, std={real.std():.6f}")
    print(f"FAKE : mean={fake.mean():.6f}, std={fake.std():.6f}")

    pooled_std = np.sqrt(
        (real.std() ** 2 + fake.std() ** 2) / 2
    )

    if pooled_std > 0:
        effect = abs(real.mean() - fake.mean()) / pooled_std
    else:
        effect = 0

    print(f"Effect size: {effect:.4f}")

print("\n" + "=" * 70)
print("CORRELATION WITH REAL/FAKE LABEL")
print("=" * 70)

df["target"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .eq("fake")
    .astype(int)
)

correlations = []

for feature in FEATURE_COLUMNS:
    corr = df[feature].corr(df["target"])
    correlations.append((feature, abs(corr), corr))

correlations.sort(key=lambda x: x[1], reverse=True)

for feature, abs_corr, corr in correlations:
    print(f"{feature:25s} correlation={corr:+.4f}")

print("\n" + "=" * 70)
print("FEATURES BY GENERATOR")
print("=" * 70)

for feature in FEATURE_COLUMNS:
    print(f"\n--- {feature} ---")

    table = (
        df.groupby("generator")[feature]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    print(table.to_string())

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)