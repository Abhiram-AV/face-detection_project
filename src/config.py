import torch
import os
import onnxruntime as ort  

# Determine base directory dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Dirs ---
DATA_DIR = "D:/workspace/Images"  # file path
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROCESSED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "processed_images")
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

DATA_SPLIT_FILE = os.path.join(DATA_DIR, "dataset_split.json")

# --- Dataset ---
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Model ---
PRETRAINED_MODEL_NAME = "w600k_r50"
INSIGHTFACE_MODEL_NAME = 'buffalo_l'
INSIGHTFACE_DOWNLOAD_PATH = os.path.join(os.path.expanduser("~"), ".insightface", "models")
os.makedirs(INSIGHTFACE_DOWNLOAD_PATH, exist_ok=True)

INSIGHTFACE_DET_THRESHOLD = 0.1
IMG_SIZE = (112, 112)

# --- Fine-Tuning Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = 0 if torch.cuda.is_available() else -1

# Check ONNXRuntime GPU availability for InsightFace (optional but helpful)
AVAILABLE_PROVIDERS = ort.get_available_providers()
USE_CUDA_FOR_ONNX = 'CUDAExecutionProvider' in AVAILABLE_PROVIDERS
ONNX_CTX_ID = 0 if USE_CUDA_FOR_ONNX else -1
ONNX_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_CUDA_FOR_ONNX else ['CPUExecutionProvider']

# --- Training ---
LEARNING_RATE = 1e-5
EPOCHS = 15
BATCH_SIZE = 32

# --- Evaluation ---
SIMILARITY_THRESHOLD = 0.4
