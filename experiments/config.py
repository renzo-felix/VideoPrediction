from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
ACTIVATIONS_DIR = BASE_DIR / "activations"

# Models
VJEPA_CODE_PATH = BASE_DIR.parent / "vjepa2"
VJEPA_CHECKPOINT = Path("/home/renzo.felix/VideoPrediction/vjepa2/checkpoints/vitl.pt")
VJEPA_LAYER = 9
VJEPA_HIDDEN_DIM = 1024

VIDEOMAE_CODE_PATH = BASE_DIR.parent / "VideoMAEv2_Proyecto_Paper"
VIDEOMAE_CHECKPOINT = Path("/home/renzo.felix/VideoPrediction/vjepa2/checkpoints/videomae_vitg.pth")
VIDEOMAE_LAYER = 9
VIDEOMAE_HIDDEN_DIM = 1408

IMG_SIZE = 224
NUM_FRAMES = 16
TUBELET_SIZE = 2
PATCH_SIZE = 16

# Video generation
SIMULATION_STEPS = 240
FPS = 30
VIDEO_LENGTH = 1.0
CAMERA_DISTANCE = 5.0
CAMERA_YAW = 0
CAMERA_PITCH = -30
CAMERA_TARGET = [0, 0, 0]

OBJECT_PROPERTIES = {
    'shape': ['sphere', 'cube', 'cylinder', 'capsule'],
    'color': ['red', 'blue', 'green', 'yellow', 'white', 'black'],
    'size': ['small', 'medium', 'large', 'xlarge'],
    'material': ['matte', 'glossy', 'metallic'],
    'texture': ['plain', 'checkerboard', 'stripes']
}

SIZE_MAPPING = {
    'small': 0.25,
    'medium': 0.5,
    'large': 0.75,
    'xlarge': 1.0
}

COLOR_MAPPING = {
    'red': [1.0, 0.0, 0.0, 1.0],
    'blue': [0.0, 0.0, 1.0, 1.0],
    'green': [0.0, 1.0, 0.0, 1.0],
    'yellow': [1.0, 1.0, 0.0, 1.0],
    'white': [1.0, 1.0, 1.0, 1.0],
    'black': [0.1, 0.1, 0.1, 1.0]
}

MATERIAL_MAPPING = {
    'matte': [0.0, 0.0, 0.0],
    'glossy': [0.5, 0.5, 0.5],
    'metallic': [1.0, 1.0, 1.0]
}

SPEED_RANGE = (0, 50)
N_VIDEOS_PER_CONDITION = 20

# Steering
STEERING_LOW_PERCENTILE = 33
STEERING_HIGH_PERCENTILE = 67
NORMALIZE_STEERING = True

# Testing
TEST_CATEGORIES = ['size', 'color', 'shape', 'material', 'texture']
ALPHA = 0.05

# Compute
DEVICE = 'cpu'
BATCH_SIZE = 1
N_WORKERS = 8
CACHE_ACTIVATIONS = False

# Files
TEST_METADATA_TEMPLATE = "metadata/test/test_metadata_{category}.csv"
