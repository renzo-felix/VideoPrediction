import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import *

def load_videomae(checkpoint_path, device='cpu'):
    if not VIDEOMAE_CODE_PATH.exists():
        raise FileNotFoundError(f"VideoMAE code not found at {VIDEOMAE_CODE_PATH}")

    sys.path.insert(0, str(VIDEOMAE_CODE_PATH))

    from modeling_pretrain import PretrainVisionTransformerEncoder

    model = PretrainVisionTransformerEncoder(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        in_chans=3,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=torch.nn.LayerNorm,
        init_values=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' not in checkpoint:
        raise ValueError(f"Invalid VideoMAE checkpoint: missing 'model' key")

    cleaned = {k.replace("module.", "").replace("encoder.", ""): v for k, v in checkpoint['model'].items()}

    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model
