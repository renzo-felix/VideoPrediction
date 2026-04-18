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

    activations = []
    speeds = []
    video_ids = []

    for idx, row in df.iterrows():
        video_id = row['video_id']
        act_path = layer_dir / f"{video_id}.npy"

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

    logger.info(f"Loaded {len(speeds)} activations")
    logger.info(f"Speed range: [{speeds.min():.2f}, {speeds.max():.2f}]")

    return activations, speeds, video_ids, df
