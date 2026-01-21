import os
import shutil
import time
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torch.onnx

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentinel_train")

# Constants
SAMPLE_RATE = 22050
DURATION = 4  # seconds
TARGET_SAMPLE_COUNT = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 10
FOLDS = 10
CLASS_NAMES = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
    'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]

# Paths (Assumes running from repo root or configured via env vars)
# In production/local, data should be in ./urbansound8k
DATA_ROOT = os.environ.get("DATA_ROOT", "./urbansound8k")
METADATA_PATH = os.path.join(DATA_ROOT, 'metadata', 'UrbanSound8K.csv')
AUDIO_DIR = os.path.join(DATA_ROOT, 'audio')
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "./model")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class UrbanSoundDataset(Dataset):
    """
    Optimized Dataset that handles loading, resampling, mixing, and padding.
    """
    def __init__(self, df, audio_dir, target_sample_rate, num_samples, cache_to_ram=True):
        self.df = df
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.cache_to_ram = cache_to_ram
        self.cache = {}

        n_rows = self.df.shape[0]
        logger.info(f"Dataset initialized with {n_rows} files.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.cache_to_ram and index in self.cache:
            return self.cache[index]

        row = self.df.iloc[index]
        # Handle slice_file_name being potentially different in CSV vs Folders
        audio_path = os.path.join(self.audio_dir, f"fold{row['fold']}", row['slice_file_name'])
        label = row['classID']

        try:
            signal, sr = torchaudio.load(audio_path)

            if sr != self.target_sample_rate:
                resampler = T.Resample(sr, self.target_sample_rate)
                signal = resampler(signal)

            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            length_signal = signal.shape[1]
            if length_signal > self.num_samples:
                signal = signal[:, :self.num_samples]
            elif length_signal < self.num_samples:
                num_missing = self.num_samples - length_signal
                signal = torch.nn.functional.pad(signal, (0, num_missing))

            result = (signal, label)

            if self.cache_to_ram:
                self.cache[index] = result

            return result

        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            dummy_signal = torch.zeros((1, self.num_samples))
            return dummy_signal, label

# Transforms
transform = nn.Sequential(
    T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    ),
    T.AmplitudeToDB()
)

augment = nn.Sequential(
    T.FrequencyMasking(freq_mask_param=10),
    T.TimeMasking(time_mask_param=35)
)

class AudioClassifierDropout(nn.Module):
    def __init__(self, num_classes, transform_block, augmentation_block=None, dropout_prob=0.5):
        super().__init__()
        self.transform = transform_block
        self.augment = augmentation_block

        self.bn0 = nn.BatchNorm2d(1)
        self.model = resnet18(weights='IMAGENET1K_V1')
        # Modify first conv to accept 1 channel (spectrogram) instead of 3 (RGB)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features
        # Add Dropout
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        x = self.transform(x)
        x = self.bn0(x)

        if self.training and self.augment is not None:
            x = self.augment(x)

        return self.model(x)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return running_loss / len(dataloader), accuracy, f1

def export_to_onnx(model, device):
    """Exports the trained model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 1, int(TARGET_SAMPLE_COUNT)).to(device)
    onnx_filename = "echo_audio_clf.onnx"
    save_path = os.path.join(MODEL_SAVE_DIR, onnx_filename)
    
    logger.info(f"Exporting model to {save_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['waveform'],
        output_names=['class_logits'],
        dynamic_axes={'waveform': {0: 'batch_size', 2: 'samples'}, 'class_logits': {0: 'batch_size'}}
    )
    logger.info("ONNX Export Successful.")

def main():
    if not os.path.exists(METADATA_PATH):
        logger.error(f"Metadata not found at {METADATA_PATH}. Please ensure data is mounted or downloaded.")
        return

    logger.info("Loading Metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    
    # Using Fold 1 as Validation, others as Train (Simplified for this script examples)
    # Ideally should loop through folds if doing full CV locally
    fold_idx = 1
    train_df = metadata[metadata['fold'] != fold_idx]
    val_df = metadata[metadata['fold'] == fold_idx]

    train_ds = UrbanSoundDataset(train_df, AUDIO_DIR, SAMPLE_RATE, TARGET_SAMPLE_COUNT)
    val_ds = UrbanSoundDataset(val_df, AUDIO_DIR, SAMPLE_RATE, TARGET_SAMPLE_COUNT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for compatibility
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using Device: {device}")

    # OPTIMAL HYPERPARAMS from echo.py
    LR = 0.0001
    DROPOUT_PROB = 0.7
    WEIGHT_DECAY = 1e-4
    
    model = AudioClassifierDropout(NUM_CLASSES, transform, augment, dropout_prob=DROPOUT_PROB).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Class Weights
    logger.info("Computing Class Weights...")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['classID']), y=train_df['classID'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    logger.info(f"Starting Training with LR={LR}, Dropout={DROPOUT_PROB}, WD={WEIGHT_DECAY}")
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    
    # Early Stopping
    patience_counter = 0
    early_stopping_patience = 3

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Check for improvement (echo.py uses F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            
            save_path = os.path.join(MODEL_SAVE_DIR, 'sentinel_best.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"New Best F1! Model saved to {save_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping triggered.")
            break

    # Export
    # Reload best model before export?
    # echo.py reloads it. It's safer.
    if os.path.exists(os.path.join(MODEL_SAVE_DIR, 'sentinel_best.pth')):
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, 'sentinel_best.pth')))
        
    export_to_onnx(model, device)

if __name__ == "__main__":
    main()

