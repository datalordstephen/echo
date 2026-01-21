# Sentinel - Multi-Class Audio Classifier

Sentinel is a deep learning-based audio classification system designed to identify urban sounds. It leverages **ResNet18** and **Mel-Spectrograms** to achieve high accuracy on the **UrbanSound8K** dataset. The project is refactored for production, featuring a reduced-dependency inference pipeline and a Dockerized API ready for **AWS Lambda**.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [EDA](#eda)
- [Modelling](#modelling)
- [Deployment/Containerization](#deploymentcontainerization)
- [API Usage](#api-usage)

---

## Problem Statement
Urban noise pollution is a growing environmental concern. Automatically classifying sounds (e.g., jackhammers, sirens, playing children) can enable smart city monitoring systems to assess noise levels and sources in real-time. This project aims to build a robust classifier that can take raw audio input and output the sound category with high confidence.

## Dataset
The project uses the **UrbanSound8K** dataset, which contains **8,732 labeled sound excerpts** (<= 4 seconds) from 10 classes:
1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Bark
5. Drilling
6. Engine Idling
7. Gun Shot
8. Jackhammer
9. Siren
10. Street Music

The data is pre-sorted into 10 folds for cross-validation.

## Project Structure
The repository follows a clean, production-ready structure managed by `uv`:

```
sentinel/
├── model/                     # Stores ONNX models and checkpoints
│   └── echo_audio_clf.onnx    # Final exported quantization-ready model
├── src/                       # Source code
│   ├── api.py                 # FastAPI application & Lambda Handler
│   ├── predict.py             # Inference logic (ONNXRuntime + Librosa)
│   └── train.py               # Training pipeline (PyTorch)
├── Dockerfile                 # AWS Lambda-optimized Docker image
├── pyproject.toml             # Dependency management (Base vs Training)
├── uv.lock                    # Exact dependency versions
└── README.md                  # Project documentation
```

## EDA (Exploratory Data Analysis)
Key insights driven by data analysis:
*   **Class Imbalance**: Some classes (e.g., *car_horn*, *gun_shot*) have fewer samples. We implemented **Class Weighting** (`balanced`) in the CrossEntropyLoss to mitigate this.
*   **Variable Lengths**: Audio clips vary from <1s to 4s. The preprocessing pipeline standardizes inputs by padding (silence) or truncating to exactly **4 seconds**.
*   **Spectrograms**: Visual inspection of Mel-Spectrograms showed distinct patterns for classes like *siren* (continuous lines) vs *jackhammer* (repetitive bursts), confirming CNNs as a viable architecture.

## Modelling
We treat audio classification as an image classification problem using **Transfer Learning**.

*   **Input**: Raw Audio Waveform -> Log Mel-Spectrogram (1 Channel).
*   **Preprocessing**: 
    *   Resample to 22.05kHz.
    *   Convert Stereo to Mono.
    *   Generate MelSpectrogram (128 bands).
    *   Convert to Decibels (Log scale).
*   **Augmentation** (Training only):
    *   **Frequency Masking**: Randomly masks frequency bands.
    *   **Time Masking**: Randomly masks time steps.
*   **Architecture**:
    *   **Backbone**: **ResNet18** (Pretrained on ImageNet).
    *   **Modification**: First Convolutional layer modified to accept 1 channel (instead of 3).
    *   **Head**: Dropout (`p=0.7`) + Fully Connected Layer (10 outputs).
*   **Optimization**:
    *   **Optimizer**: Adam (`lr=1e-4`, `weight_decay=1e-4`).
    *   **Loss**: Weighted CrossEntropyLoss.
    *   **Early Stopping**: Monitors validation F1-score with patience of 3 epochs.

## Deployment/Containerization
The project is containerized for easy deployment on **Railway** (Recommended) or **AWS Lambda**:

1.  **Lightweight Inference**: PyTorch is stripped out of the inference container. We use `onnxruntime` and `numpy`/`librosa` for prediction.
2.  **Package Management**: Uses `uv` for extremely fast resolution and installation.
3.  **Docker**:
    *   Base Image: `python:3.12-slim`.
    *   Installs system libs (`libsndfile`).
    *   Syncs only production dependencies from `uv.lock`.
    *   Runs `uvicorn` on `$PORT`.

**Railway Deployment:**
Simply connect this repository to Railway. It will automatically detect the Dockerfile and deploy.

**Build Command (Local):**
```bash
docker build -t sentinel-app .
```

## API Usage
The application exposes a REST API via FastAPI.

### Endpoint: Predict Audio
**POST** `/predict`

**Request:** `multipart/form-data`
- `file`: The audio file (`.wav`) to classify.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/my_audio.wav"
```

**Example Response:**
```json
{
  "class": "dog_bark",
  "class_id": 3,
  "confidence": 0.98
}
```

### Endpoint: Health Check
**GET** `/health`
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
