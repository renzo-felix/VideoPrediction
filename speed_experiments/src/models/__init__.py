from .base import ActivationExtractor
from .vjepa import load_vjepa
from .videomae import load_videomae

def preprocess_frames(frames, model_name):
    return frames.permute(0, 2, 1, 3, 4)

__all__ = ['ActivationExtractor', 'load_vjepa', 'load_videomae', 'preprocess_frames']
