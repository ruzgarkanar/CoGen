import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'microsoft/deberta-v3-large-squad')
ENCODER_MODEL = os.getenv('ENCODER_MODEL', 'sentence-transformers/all-mpnet-base-v2')
HF_TOKEN = os.getenv('HF_TOKEN')

LEARNING_RATE = float(os.getenv('LEARNING_RATE', 5e-6))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 50))
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE', 1))
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
WARMUP_RATIO = float(os.getenv('WARMUP_RATIO', 0.1))
MAX_GRAD_NORM = float(os.getenv('MAX_GRAD_NORM', 0.1))
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', 5))

TOP_K_RETRIEVAL = int(os.getenv('TOP_K_RETRIEVAL', 3))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))

ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
CACHE_EXPIRY = int(os.getenv('CACHE_EXPIRY', 3600))
MAX_CACHE_ITEMS = int(os.getenv('MAX_CACHE_ITEMS', 1000))

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'password': os.getenv('REDIS_PASSWORD', None),
}

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))
VECTOR_DIMENSION = 768 

NUM_WORKERS = int(os.getenv('NUM_WORKERS', 4))
DEVICE = os.getenv('DEVICE', 'cpu')  

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
