import sys
import torch
from pathlib import Path
import importlib.util

_config_path = Path(__file__).parent.parent.parent / 'config.py'
_spec = importlib.util.spec_from_file_location("config", _config_path)
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)
VIDEOMAE_CODE_PATH = _config.VIDEOMAE_CODE_PATH
IMG_SIZE = _config.IMG_SIZE
PATCH_SIZE = _config.PATCH_SIZE
NUM_FRAMES = _config.NUM_FRAMES
TUBELET_SIZE = _config.TUBELET_SIZE

def load_videomae(checkpoint_path, device='cpu'):
    if not VIDEOMAE_CODE_PATH.exists():
        raise FileNotFoundError(f"VideoMAE code not found at {VIDEOMAE_CODE_PATH}")

    _parent = str(Path(__file__).parent.parent.parent)
    _was_in_path = _parent in sys.path

    if _was_in_path:
        sys.path.remove(_parent)

    videomae_parent = str(VIDEOMAE_CODE_PATH.parent)
    sys.path.insert(0, videomae_parent)

    try:
        from VideoMAEv2_Proyecto_Paper.models.modeling_pretrain import PretrainVisionTransformerEncoder

        model = PretrainVisionTransformerEncoder(
            img_size=IMG_SIZE,
            patch_size=14,
            all_frames=NUM_FRAMES,
            tubelet_size=TUBELET_SIZE,
            in_chans=3,
            embed_dim=1408,
            depth=40,
            num_heads=16,
            mlp_ratio=48/11,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            init_values=0.0,
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['module']
        cleaned = {k.replace("module.", "").replace("encoder.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(cleaned, strict=False)
        model.to(device).eval()
        return model
    finally:
        if _was_in_path:
            sys.path.insert(0, _parent)
