import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pandas as pd
import argparse
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from data import VideoDataset
from models import load_vjepa, load_videomae, ActivationExtractor, preprocess_frames

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_activations(metadata_path, output_dir, model_name, checkpoint_path, layer, device, batch_size, videos_base_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / f"layer_{layer}"
    layer_dir.mkdir(exist_ok=True)

    dataset = VideoDataset(metadata_path, videos_base_dir, IMG_SIZE, NUM_FRAMES)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"\nLoading {model_name} model...")
    device_obj = torch.device(device)

    MODEL_LOADERS = {'vjepa': load_vjepa, 'videomae': load_videomae}
    if model_name not in MODEL_LOADERS:
        raise ValueError(f"Unknown model: {model_name}")

    model = MODEL_LOADERS[model_name](checkpoint_path, device_obj)
    extractor = ActivationExtractor(model, layer, model_name)

    all_metadata = []
    for batch in tqdm(dataloader, desc="Processing batches"):
        frames = batch['frames'].to(device_obj)
        frames = preprocess_frames(frames, model_name)
        activations = extractor(frames)

        for i, video_id in enumerate(batch['video_id']):
            act = activations[layer][i]
            np.save(layer_dir / f"{video_id}.npy", act.numpy())
            all_metadata.append({
                'video_id': video_id,
                'actual_speed': batch['actual_speed'][i].item(),
                'target_speed': batch['target_speed'][i].item(),
            })

    pd.DataFrame(all_metadata).to_csv(output_dir / "activation_metadata.csv", index=False)
    extractor.remove_hooks()
    logger.info("EXTRACTION COMPLETE")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["training", "test"])
    parser.add_argument("--category", choices=TEST_CATEGORIES)
    parser.add_argument("--model", default='vjepa', choices=['vjepa', 'videomae'])
    parser.add_argument("--checkpoint")
    parser.add_argument("--layer", type=int, default=VJEPA_LAYER)
    parser.add_argument("--device", default=DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "test" and not args.category:
        parser.error("--category required when mode=test")

    if args.checkpoint is None:
        args.checkpoint = VJEPA_CHECKPOINT if args.model == 'vjepa' else VIDEOMAE_CHECKPOINT

    if args.mode == "training":
        metadata_path = DATA_DIR / TRAINING_METADATA
        output_dir = ACTIVATIONS_DIR / f"training_{args.model}"
    else:
        metadata_path = DATA_DIR / TEST_METADATA_TEMPLATE.format(category=args.category)
        output_dir = ACTIVATIONS_DIR / f"test_{args.category}_{args.model}"

    extract_activations(metadata_path, output_dir, args.model, args.checkpoint, args.layer, args.device, args.batch_size, BASE_DIR)

if __name__ == "__main__":
    main()
