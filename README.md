# Echo: Intelligent Urban Sound Classification

Echo is a high-performance, production-ready audio classification system designed to automatically detect and categorize environmental noise in urban settings. Built with PyTorch and optimized via ONNX, Echo provides a scalable solution for real-time acoustic monitoring.

The project is designed for **Scalability** and **Ease of Deployment**, featuring:
*   **Modular Architecture**: Split into `train`, `predict`, and `api`.
*   **Fast Inference**: Uses `ONNX Runtime` and `Librosa` (No PyTorch dependency in production).
*   **Modern Tooling**: Managed by `uv` for lightning-fast dependency resolution.
*   **Containerization**: Docker-ready for **Railway** (Recommended) or AWS Lambda.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Detailed Setup & Installation](#detailed-setup--installation)
- [Usage (Training & Inference)](#usage-training--inference)
- [Dockerization & Deployment](#dockerization--deployment)
- [API Documentation](#api-documentation)

---

## Problem Statement
In the rapidly growing urban landscape, noise pollution has become a critical environmental concern affecting public health, safety, and urban planning. Monitoring these acoustic environments manually is labor-intensive and impossible to scale.

City planners, security agencies, and environmental researchers need an automated way to:

Identify specific noise sources: Differentiating between harmless "street music" and critical "sirens" or "gunshots."

Monitor Noise Pollution: Mapping the intensity and frequency of industrial sounds like "drilling" or "jackhammers."

Trigger Real-time Responses: Enabling smart city infrastructure to react instantly when specific emergency sounds are detected.

Echo solves these challenges by providing a robust deep-learning pipeline that converts raw audio into actionable data, optimized for low-latency deployment in serverless environments.

## Dataset
**UrbanSound8K**: 8,732 labeled sound excerpts (<= 4s) from 10 classes:
*   `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`
*   `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`

## Project Structure
```
echo/
├── model/                     # Stores ONNX models and checkpoints
│   └── echo_audio_clf.onnx    # Optimized Inference Model
├── src/                       # Source code
│   ├── api.py                 # FastAPI Application
│   ├── predict.py             # Inference Logic (ONNX + Librosa)
│   └── train.py               # Training Pipeline (PyTorch)
├── Dockerfile                 # Production Docker Image
├── pyproject.toml             # Dependencies (Base vs Training)
├── uv.lock                    # Locked Checksums
└── README.md                  # Documentation
```

---

## Detailed Setup & Installation

Follow these steps to set up the project locally.

### 1. Prerequisites
*   **Python 3.12+**
*   **uv** (Package Manager): [Install Guide](https://github.com/astral-sh/uv)
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/echo.git
cd echo
```

### 3. Install Dependencies
We use `uv` to manage two sets of dependencies:
*   **Base**: Lightweight, for running the API/Inference.
*   **Training**: Heavy, includes PyTorch (for training only).

**To install EVERYTHING (for development/training):**
```bash
uv sync --extra training
uv sync # development only
```

**Activate the Virtual Environment:**
```bash
source .venv/bin/activate
```

---

## Usage (Training & Inference)

### 1. Prepare Data
We use `soundata` to automatically download and validate the **UrbanSound8K** dataset.

Run the download script:
```bash
python src/download_data.py
```
This will download the dataset to `urbansound8k/` in the project root.

### 2. Train the Model
This will run the training loop, validate on the hold-out fold, and export the generic ONNX model to `model/echo_audio_clf.onnx`.
```bash
python src/train.py
```

### 3. Run Inference (CLI)
You can test predictions using the helper script:
```bash
# Edit the script to point to your .wav file
python src/predict.py
```

### 4. Run the API Locally
Start the FastAPI server using Uvicorn:
```bash
uvicorn src.api:app --reload
```
Test it at `http://localhost:8000/docs`.

---

## Dockerization & Deployment

### Run with Docker (Locally)
The strict separation of `base` vs `training` dependencies ensures our Docker image is small (no PyTorch).

1.  **Build the Image:**
    ```bash
    docker build -t echo-app .
    ```

2.  **Run the Container:**
    ```bash
    docker run -p 8000:8000 -e PORT=8000 echo-app
    ```

3.  **Test:**
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@test.wav"
    ```

### Deploy to Railway (Recommended)
This project is configured for zero-config deployment on [Railway](https://railway.app/).

1.  Push this code to a GitHub repository.
2.  Login to Railway and click **"New Project"**.
3.  Select **"Deploy from GitHub repo"** and choose `echo`.
4.  Railway detects the `Dockerfile` and deploys automatically.

---

## API Documentation

### POST `/predict`
Uploads an audio file for classification.

**Request:** `multipart/form-data`
*   `file`: `.wav` audio file.

**Response:**
```json
{
  "class": "siren",
  "class_id": 8,
  "confidence": 0.99
}
```

### GET `/health`
Health check endpoint.
```json
{
  "status": "healthy",
  "model_loaded": true
}
```
