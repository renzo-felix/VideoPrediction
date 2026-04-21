import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_activations(activations_dir, metadata_path, layer):
    activations_dir = Path(activations_dir)
    layer_dir = activations_dir / f"layer_{layer}"

    if not layer_dir.exists():
        raise ValueError(f"Layer directory not found: {layer_dir}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata: {len(df)} videos")

    DEFAULTS = {
        'size': ('sphere', 'medium', 'red', 'matte', 'plain'),
        'color': ('sphere', 'medium', 'red', 'matte', 'plain'),
        'shape': ('sphere', 'medium', 'red', 'matte', 'plain'),
        'material': ('sphere', 'medium', 'red', 'matte', 'plain'),
        'texture': ('sphere', 'medium', 'red', 'matte', 'plain'),
    }

    activations = []
    speeds = []
    video_ids = []

    for idx, row in df.iterrows():
        video_id = row['video_id']
        act_path = layer_dir / f"{video_id}.npy"

        if not act_path.exists() and 'category' in row and 'condition' in row:
            number = video_id.rsplit('_', 1)[-1]
            category = row['category']
            variation = row['condition']

            if category in DEFAULTS:
                shape, size, color, material, texture = DEFAULTS[category]
                if category == 'size':
                    size = variation
                elif category == 'color':
                    color = variation
                elif category == 'shape':
                    shape = variation
                elif category == 'material':
                    material = variation
                elif category == 'texture':
                    texture = variation

                full_video_id = f"{category}_{shape}_{size}_{color}_{material}_{texture}_{number}"
                act_path = layer_dir / f"{full_video_id}.npy"

        if act_path.exists():
            act = np.load(act_path)
            if act.ndim == 2:
                act = act.mean(axis=0)
            activations.append(act)
            speeds.append(row['actual_speed'])
            video_ids.append(video_id)
        else:
            logger.warning(f"Activation not found: {act_path}")

    activations = np.array(activations)
    speeds = np.array(speeds)

    if len(speeds) > 0:
        logger.info(f"Loaded {len(speeds)} activations")
        logger.info(f"Speed range: [{speeds.min():.2f}, {speeds.max():.2f}]")
    else:
        logger.error(f"No activations found in {layer_dir}")

    return activations, speeds, video_ids, df
