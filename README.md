# Deepfake Detection Using Spatial and Temporal Features

An AI-powered deepfake detection system designed to identify manipulated images and videos using a combination of deep learning architectures and traditional machine learning techniques.

## Overview

This project explores multiple approaches to deepfake detection by comparing different feature extraction techniques and classification models.

Rather than relying on a single architecture, the project investigates both deep learning and conventional machine learning approaches to understand how different models perform when distinguishing authentic media from manipulated content.

The project covers:
- Image-based deepfake detection.
- Video-based deepfake detection.
- Spatial feature extraction and classification.
- Comparative evaluation of deep learning and traditional ML models.
- Performance analysis across different architectures.

## Models and Algorithms

### Deep Learning Models

| Model | Approach | Status |
|---|---|---|
| EfficientNet-B0 | Spatial feature extraction | Existing implementation |
| Xception | Deepfake classification | Existing implementation |
| ConvNeXt-Tiny | CNN-based feature learning | Planned / In progress |
| ViT-Base | Transformer-based visual feature learning | Planned / In progress |

### Traditional Machine Learning

| Algorithm | Feature Extraction | Status |
|---|---|---|
| HOG + SVM | Histogram of Oriented Gradients | Planned / In progress |
| LBP + XGBoost | Local Binary Patterns | Planned / In progress |
| XGBoost | Conventional ML classification baseline | Planned / In progress |

The conventional ML approaches provide a comparison against deep learning architectures, helping evaluate whether handcrafted visual features can effectively distinguish manipulated content.

## System Architecture

The proposed system follows a modular pipeline:

```text
             Input Media
                  |
          +-------+-------+
          |               |
        Image           Video
          |               |
     Preprocessing   Frame Extraction
          |               |
          |          Frame Preprocessing
          |               |
          +-------+-------+
                  |
          Feature Extraction
                  |
       +----------+----------+
       |                     |
  Deep Learning         Traditional ML
       |                     |
 EfficientNet-B0          HOG + SVM
 Xception                 LBP + XGBoost
 ConvNeXt-Tiny
 ViT-Base
       |                     |
       +----------+----------+
                  |
          Classification
                  |
       Real / Fake Prediction
                  |
         Performance Analysis
```

## Video Detection Pipeline

The video detection component extends image-based classification to video inputs.

The pipeline involves:
1. Accepting a video file as input.
2. Extracting frames from the video.
3. Preprocessing the extracted frames.
4. Passing frames through the selected detection models.
5. Aggregating frame-level predictions.
6. Producing a video-level classification result.

The video pipeline aims to investigate how frame-level predictions can be used to identify manipulated video content. Frame-level aggregation does not, by itself, constitute temporal modelling; explicit temporal features or sequence models would be needed for that.

## Performance Evaluation

Models will be evaluated using standard classification metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive predictions |
| Recall | Ability to identify manipulated samples |
| F1-Score | Balance between precision and recall |
| Confusion Matrix | Distribution of correct and incorrect predictions |
| Inference Time | Time required to generate predictions |

Where applicable, computational requirements and model complexity will also be considered.

## Technology Stack

**Programming Language**
- Python

**Deep Learning**
- PyTorch
- Torchvision
- Timm

**Machine Learning**
- Scikit-learn
- XGBoost
- OpenCV

**Data Processing and Visualization**
- NumPy
- Pandas
- Matplotlib
- Seaborn

**Experimentation**
- Google Colab
- MLflow (for experiment tracking, if configured)

**Deployment**
- Streamlit (planned interface)

## Project Structure

The following is a suggested structure. Update it to reflect the actual repository layout.

```text
Deepfake-Detection/
│
├── datasets/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── efficientnet_experiments.ipynb
│   ├── xception_experiments.ipynb
│   ├── convnext_experiments.ipynb
│   ├── vit_experiments.ipynb
│   └── traditional_ml.ipynb
│
├── models/
│   ├── checkpoints/
│   └── trained_models/
│
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── image_detection/
│   ├── video_detection/
│   └── evaluation/
│
├── app/
│   └── streamlit_app.py
│
├── results/
│   ├── metrics/
│   └── visualizations/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Deepfake-Detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The project uses image and video datasets for training and evaluation.

Document the dataset names, sources, preprocessing procedures, and train-validation-test splits here once they are finalized. Dataset files and trained model checkpoints are not included in the repository unless explicitly specified.

## Future Work

- Complete comparative experiments across the selected architectures.
- Integrate image and video detection into a unified pipeline.
- Investigate temporal information across video frames.
- Improve video-level prediction aggregation.
- Compare deep learning approaches against handcrafted feature-based ML models.
- Develop an interactive Streamlit interface.
- Consolidate experimental results into a comparative evaluation report.

## Limitations

- Detection performance depends on the quality and diversity of the training datasets.
- Models may generalize differently to unseen manipulation techniques.
- Frame-based video classification may not capture all temporal inconsistencies.
- Predictions should be treated as model-generated assessments rather than definitive proof of manipulation.

## Disclaimer

This project is developed for academic and educational purposes, focusing on deepfake detection, computer vision, and comparative machine learning experimentation.

It is not intended to serve as a definitive forensic verification system.
