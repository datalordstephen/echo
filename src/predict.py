import os
import onnxruntime as ort
import numpy as np
import librosa
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentinel_predict")

CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
    'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]

SAMPLE_RATE = 22050
DURATION = 4
TARGET_SAMPLES = SAMPLE_RATE * DURATION

class SentinelPredictor:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading ONNX Model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, file_path):
        """
        Loads and preprocesses audio using Librosa/Numpy (No Torch).
        Matches the steps in training: Load -> Resample -> Mono -> Pad/Truncate
        """
        try:
            # Load audio using librosa (already handles resampling if sr is provided)
            # librosa loads as (n_channels, n_samples) if mono=False, or (n_samples,) if mono=True
            # We want mono, so mono=True is default but let's be explicit manually
            signal, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            
            # Ensure shape is (1, length) for consistency
            signal = signal[np.newaxis, :] 

            # Pad or Truncate
            length_signal = signal.shape[1]
            if length_signal > TARGET_SAMPLES:
                signal = signal[:, :TARGET_SAMPLES]
            elif length_signal < TARGET_SAMPLES:
                pad_width = TARGET_SAMPLES - length_signal
                # Pad last dim (samples) with zeros on the right
                signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')

            # Expand dims for batch: (1, 1, samples)
            signal = signal[np.newaxis, :, :]
            
            return signal.astype(np.float32)

        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {e}")
            return None

    def predict_single(self, file_path):
        """
        Predicts the class of a single audio file.
        Returns: (class_name, confidence, raw_logits)
        """
        input_tensor = self.preprocess(file_path)
        if input_tensor is None:
            return None

        # Run Inference
        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
        # Softmax for confidence
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        probs = softmax(logits)
        pred_idx = np.argmax(probs)
        confidence = probs[0][pred_idx]
        
        class_name = CLASS_NAMES[pred_idx]
        logger.info(f"File: {os.path.basename(file_path)} -> Predict: {class_name} ({confidence:.2f})")
        
        return {
            "class": class_name,
            "class_id": int(pred_idx),
            "confidence": float(confidence)
        }

    def predict_batch(self, directory):
        """
        Predicts all .wav files in a directory.
        """
        results = []
        files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        logger.info(f"Found {len(files)} Wav files in {directory}")

        for f in files:
            path = os.path.join(directory, f)
            res = self.predict_single(path)
            if res:
                res['file'] = f
                results.append(res)
        
        return results

if __name__ == "__main__":
    # Example Usage
    MODEL_PATH = os.path.join("model", "echo_audio_clf.onnx")
    predictor = SentinelPredictor(MODEL_PATH)
    
    # Create a dummy file for testing if none exist
    # import soundfile as sf
    # dummy_audio = np.random.uniform(-1, 1, TARGET_SAMPLES)
    # sf.write('test.wav', dummy_audio, SAMPLE_RATE)
    # predictor.predict_single('test.wav')
