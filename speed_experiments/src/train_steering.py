import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import argparse
import logging

from config import *
from analysis import load_activations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_steering_vector(activations, values, low_percentile=33, high_percentile=67):
    low_threshold = np.percentile(values, low_percentile)
    high_threshold = np.percentile(values, high_percentile)

    low_mask = values <= low_threshold
    high_mask = values >= high_threshold

    low_activations = activations[low_mask]
    high_activations = activations[high_mask]

    steering_vector = high_activations.mean(axis=0) - low_activations.mean(axis=0)
    steering_vector_normalized = steering_vector / np.linalg.norm(steering_vector)

    low_mean = low_activations.mean()
    high_mean = high_activations.mean()
    pooled_std = np.sqrt((low_activations.std()**2 + high_activations.std()**2) / 2)
    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0

    low_stats = {
        'n': low_mask.sum(),
        'mean': values[low_mask].mean(),
        'std': values[low_mask].std(),
        'min': values[low_mask].min(),
        'max': values[low_mask].max()
    }

    high_stats = {
        'n': high_mask.sum(),
        'mean': values[high_mask].mean(),
        'std': values[high_mask].std(),
        'min': values[high_mask].min(),
        'max': values[high_mask].max()
    }

    return {
        'steering_vector': steering_vector,
        'steering_vector_normalized': steering_vector_normalized,
        'low_mask': low_mask,
        'high_mask': high_mask,
        'low_stats': low_stats,
        'high_stats': high_stats,
        'effect_size': cohens_d,
        'low_threshold': low_threshold,
        'high_threshold': high_threshold
    }

def train_predictor(activations, steering_vector, values):
    projections = activations @ steering_vector
    reg = LinearRegression()
    reg.fit(projections.reshape(-1, 1), values)
    predictions = reg.predict(projections.reshape(-1, 1))
    r, _ = pearsonr(predictions, values)
    r2 = reg.score(projections.reshape(-1, 1), values)
    return reg, projections, predictions, r, r2

def train_steering_vector(concept='actual_speed', model='vjepa', activations_dir=None,
                         metadata_path=None, layer=VJEPA_LAYER, output_path=None,
                         low_percentile=STEERING_LOW_PERCENTILE, high_percentile=STEERING_HIGH_PERCENTILE):
    if activations_dir is None:
        activations_dir = ACTIVATIONS_DIR / f"training_{model}"
    if metadata_path is None:
        metadata_path = DATA_DIR / TRAINING_METADATA
    if output_path is None:
        output_path = RESULTS_DIR / f"steering_vector_{concept}_{model}.pkl"

    logger.info(f"Model: {model}")
    logger.info(f"Concept: {concept}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Activations: {activations_dir}")

    activations, _, video_ids, df = load_activations(
        activations_dir, metadata_path, layer
    )

    values = df[concept].values
    steering_info = compute_steering_vector(
        activations, values, low_percentile, high_percentile
    )

    logger.info(f"Low {concept}:  n={steering_info['low_stats']['n']}, "
               f"mean={steering_info['low_stats']['mean']:.2f}")
    logger.info(f"High {concept}: n={steering_info['high_stats']['n']}, "
               f"mean={steering_info['high_stats']['mean']:.2f}")
    logger.info(f"Cohen's d: {steering_info['effect_size']:.3f}")

    reg, projections, predictions, r, r2 = train_predictor(
        activations,
        steering_info['steering_vector_normalized'],
        values
    )

    logger.info(f"Pearson r: {r:.3f}")
    logger.info(f"R²:        {r2:.3f}")

    results = {
        'model': model,
        'concept': concept,
        'layer': layer,
        'steering_vector': steering_info['steering_vector'],
        'steering_vector_normalized': steering_info['steering_vector_normalized'],
        'low_stats': steering_info['low_stats'],
        'high_stats': steering_info['high_stats'],
        'effect_size': steering_info['effect_size'],
        'low_threshold': steering_info['low_threshold'],
        'high_threshold': steering_info['high_threshold'],
        'low_percentile': low_percentile,
        'high_percentile': high_percentile,
        'predictor': reg,
        'training_correlation': r,
        'training_r2': r2,
        'training_values': values,
        'training_predictions': predictions,
        'training_video_ids': video_ids
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"✓ Saved: {output_path}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='vjepa', choices=['vjepa', 'videomae'])
    parser.add_argument("--concept", type=str, default='actual_speed')
    parser.add_argument("--layer", type=int, default=VJEPA_LAYER)
    parser.add_argument("--low_percentile", type=int, default=STEERING_LOW_PERCENTILE)
    parser.add_argument("--high_percentile", type=int, default=STEERING_HIGH_PERCENTILE)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    results = train_steering_vector(
        model=args.model,
        concept=args.concept,
        layer=args.layer,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        output_path=args.output
    )

    print(f"{args.model}/{args.concept}: r={results['training_correlation']:.3f}, R²={results['training_r2']:.3f}, d={results['effect_size']:.3f}")

if __name__ == "__main__":
    main()
