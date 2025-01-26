# Configuration parameters for the project
class Config:
    # Data parameters
    SAMPLE_RATE = 22050  # UrbanSound8K standard sample rate
    DURATION = 4  # Most sounds are 4 seconds or less
    N_MELS = 128
    
    # Model parameters
    NUM_CLASSES = 10  # UrbanSound8K has 10 classes
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    
    # Training parameters
    BATCH_SIZE = 128  # Increased for better GPU utilization
    NUM_WORKERS = 2   # Number of DataLoader workers
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 2
    
    # CUDA Optimizations
    CUDNN_BENCHMARK = True
    ALLOW_TF32 = True
    USE_AMP = True  # Automatic Mixed Precision
    
    # Paths
    # DATA_DIR = r"D:\Documenti\MSc IS\TESI_hyperbolic\sound_datasets\urbansound8k\audio"
    # METADATA_PATH = r"D:\Documenti\MSc IS\TESI_hyperbolic\sound_datasets\urbansound8k\metadata\UrbanSound8K.csv"
    DATA_DIR = "/content/mydata/audio"
    METADATA_PATH = "/content/mydata/metadata/UrbanSound8K.csv"
    PROCESSED_DATA_DIR = "data/processed"
    MODEL_SAVE_DIR = "models/saved"
    
    # Class mapping for UrbanSound8K
    CLASS_MAPPING = {
        'air_conditioner': 0,
        'car_horn': 1,
        'children_playing': 2,
        'dog_bark': 3,
        'drilling': 4,
        'engine_idling': 5,
        'gun_shot': 6,
        'jackhammer': 7,
        'siren': 8,
        'street_music': 9
    }
    
    # Training
    DEVICE = "cuda"  # or "cpu"
    RANDOM_SEED = 42 