# Deepfake Detection Prototype — PRNU / Poisson-Gaussian / High-Frequency Noise Analysis

Detects diffusion-generated / face-morphed images by checking whether their
noise statistics match what a real camera sensor produces.

## Status of this prototype

All modules have been unit-tested on synthetic images (random noise vs.
blurred) and run without errors. **Not yet tested on real face images** —
do that first before trusting any numbers.

Known caveat: `prnu_block_consistency_score()` in `src/noise_analysis/prnu.py`
compares each block against the image's own mean residual pattern, which can
trivially score near 1.0 regardless of input. Validate this on real faces
first — if it stays flat, swap in a properly externally-estimated PRNU
fingerprint (see docstring in that file for the two modes).

## Setup (Mac)

```bash
python3 -m venv deepfake-env
source deepfake-env/bin/activate
pip install -r requirements.txt
```

Face detection uses OpenCV's built-in Haar cascade (not MediaPipe — recent
MediaPipe pip builds inconsistently ship the legacy `solutions` API across
platforms/versions, so Haar cascade was chosen for reliability in this
prototype. It's less accurate on extreme poses; swap in RetinaFace/MediaPipe
Tasks API later if accuracy on the real dataset is limited by detection).

## Run order

1. **Extract your DiffFace tars** into `data/extracted/<GeneratorName>/...`

2. **Build labels.csv:**
   ```bash
   python src/preprocessing/dataset_builder.py --root data/extracted --out data/labels.csv
   ```
   Check the printed per-generator counts look right before continuing.

3. **Inspect a few crops manually** before running on the full set:
   ```bash
   python src/preprocessing/face_detector.py data/extracted/Real/<some_image>.png
   ```
   Open `face_crop_debug.png` and confirm the face is properly framed.

4. **Extract features** (start small with `--limit` to sanity check):
   ```bash
   python src/features/feature_builder.py --labels data/labels.csv --out data/features_sample.csv --limit 200
   ```
   Once that looks right, run without `--limit` for the full subset.

5. **Train the baseline model:**
   ```bash
   python src/models/train_baseline.py --features data/features.csv --model_out models_saved/baseline_xgb.pkl
   ```
   Look at the feature importances and per-generator breakdown it prints —
   this tells you which branch (PRNU/Poisson-Gaussian/HF) is actually doing
   the work, and whether performance holds up per-generator.

6. **Cross-generator generalization test** (the real test of whether this
   approach works, not just memorizes one generator):
   ```bash
   python src/evaluation/cross_generator_eval.py \
       --features data/features.csv \
       --train_generators Real DDIM SDv15_DS0.5 \
       --test_generators LDM
   ```
   Requires downloading and running feature extraction on an additional
   held-out generator tar (e.g. LDM.tar) not used in training.

7. **Run inference on a new image:**
   ```bash
   python src/models/inference.py --model models_saved/baseline_xgb.pkl --image path/to/face.jpg
   ```

## Next steps once baseline works

- If accuracy is weak, check feature importances first — a near-zero
  importance branch means that formula needs debugging before blaming the
  overall approach.
- Add the deep-fusion CNN/LSTM option only after the XGBoost baseline is
  working and you know which branches carry signal — no point building a
  neural fusion layer on top of noise.
- Test under video compression (H.264 re-encode at multiple bitrates) if you
  plan to apply this to real-world video rather than the clean dataset
  images — compression destroys a lot of the fine noise signal used here.
