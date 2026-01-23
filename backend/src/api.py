from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
import shutil
import os
import tempfile
import logging
from .predict import EchoPredictor

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("echo_api")

app = FastAPI(title="Echo Audio Classifier API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor
predictor = None
MODEL_PATH = os.environ.get("MODEL_PATH", "model/echo_audio_clf.onnx")

@app.on_event("startup")
async def startup_event():
    global predictor
    if os.path.exists(MODEL_PATH):
        try:
            predictor = EchoPredictor(MODEL_PATH)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Prediction endpoints will fail.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Create temp file to save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        logger.info(f"Processing upload: {file.filename}")
        result = predictor.predict_single(tmp_path)

        if result is None:
            raise HTTPException(status_code=400, detail="Could not process audio file.")

        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# AWS Lambda Handler
handler = Mangum(app)
