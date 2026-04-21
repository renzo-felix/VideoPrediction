import torch
from pathlib import Path
import sys
import importlib.util

_config_path = Path(__file__).parent.parent.parent / 'config.py'
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
VJEPA_CODE_PATH = _config.VJEPA_CODE_PATH

def load_vjepa(checkpoint_path, device='cpu'):
    _parent = str(Path(__file__).parent.parent.parent)
    _was_in_path = _parent in sys.path

    if _was_in_path:
        sys.path.remove(_parent)

    try:
        encoder, _ = torch.hub.load(str(VJEPA_CODE_PATH), 'vjepa2_vit_giant', source='local', pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['encoder']
        cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

        encoder.load_state_dict(cleaned, strict=False)
        encoder.to(device).eval()
        return encoder
    finally:
        if _was_in_path:
            sys.path.insert(0, _parent)
