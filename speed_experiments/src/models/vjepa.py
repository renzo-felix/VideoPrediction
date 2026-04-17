import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import *

def load_vjepa(checkpoint_path, device='cpu'):
    encoder, _ = torch.hub.load(str(VJEPA_CODE_PATH), 'vjepa2_vit_giant', source='local', pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('encoder', checkpoint)

    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

    encoder.load_state_dict(cleaned, strict=False)
    encoder.to(device).eval()
    return encoder
