import torch
from pathlib import Path
import sys
import importlib.util

_config_path = Path(__file__).parent.parent.parent / 'config.py'
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
VJEPA_CODE_PATH = _config.VJEPA_CODE_PATH

def _detect_model_size(checkpoint_path):
    """Detecta el tamaño del modelo desde el checkpoint mirando hidden_dim."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('encoder', ckpt)
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    hidden_dim = cleaned['patch_embed.proj.weight'].shape[0]
    if hidden_dim == 1024:
        return 'vjepa2_vit_large'
    elif hidden_dim == 1280:
        return 'vjepa2_vit_huge'
    else:  # 1408
        return 'vjepa2_vit_giant'

def load_vjepa(checkpoint_path, device='cpu'):
    _parent = str(Path(__file__).parent.parent.parent)
    _was_in_path = _parent in sys.path

    if _was_in_path:
        sys.path.remove(_parent)

    try:
        hub_fn = _detect_model_size(checkpoint_path)
        encoder, _ = torch.hub.load(str(VJEPA_CODE_PATH), hub_fn, source='local', pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('encoder', checkpoint)
        cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

        encoder.load_state_dict(cleaned, strict=False)
        encoder.to(device).eval()
        return encoder
    finally:
        if _was_in_path:
            sys.path.insert(0, _parent)
