import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
import json
from scipy.stats import pearsonr, spearmanr
import argparse
import logging

from config import *
from analysis import load_activations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_activations(category, model='vjepa', layer=VJEPA_LAYER):
    activations_dir = ACTIVATIONS_DIR / f"test_{category}_{model}"
    metadata_path = DATA_DIR / f"test_metadata_{category}.csv"
    return load_activations(activations_dir, metadata_path, layer)

def test_category_transfer(steering_vector, predictor, category, model='vjepa', layer=VJEPA_LAYER):
    logger.info(f"\nTesting transfer on {category} category...")

    activations, speeds, video_ids, df = load_test_activations(category, model, layer)

    projections = activations @ steering_vector
    predictions = predictor.predict(projections.reshape(-1, 1))

    pearson_r, pearson_p = pearsonr(predictions, speeds)
    spearman_r, spearman_p = spearmanr(predictions, speeds)

    mae = np.mean(np.abs(predictions - speeds))
    rmse = np.sqrt(np.mean((predictions - speeds)**2))

    results = {
        'category': category,
        'n_videos': len(speeds),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'speed_mean': speeds.mean(),
        'speed_std': speeds.std(),
        'speed_min': speeds.min(),
        'speed_max': speeds.max(),
        'prediction_mean': predictions.mean(),
        'prediction_std': predictions.std(),
        'video_ids': video_ids,
        'speeds': speeds.tolist(),
        'predictions': predictions.tolist()
    }

    logger.info(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})")
    logger.info(f"  Spearman ρ: {spearman_r:.3f} (p={spearman_p:.2e})")
    logger.info(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return results


def test_condition_transfer(steering_vector, predictor, category, condition, model='vjepa', layer=VJEPA_LAYER):
    activations, speeds, video_ids, df = load_test_activations(category, model, layer)
    if 'condition' not in df.columns:
        raise ValueError(f"Missing 'condition' column in {category} metadata")

    mask = df['condition'] == condition
    mask_values = mask.values[:len(video_ids)]
    filtered_indices = [i for i in range(len(video_ids)) if i < len(mask_values) and mask_values[i]]

    if len(filtered_indices) == 0:
        logger.warning(f"No videos found for condition '{condition}'")
        return None

    activations_filtered = activations[filtered_indices]
    speeds_filtered = speeds[filtered_indices]

    logger.info(f"  {condition}: {len(speeds_filtered)} videos")

    projections = activations_filtered @ steering_vector
    predictions = predictor.predict(projections.reshape(-1, 1))

    pearson_r, pearson_p = pearsonr(predictions, speeds_filtered)
    spearman_r, spearman_p = spearmanr(predictions, speeds_filtered)

    mae = np.mean(np.abs(predictions - speeds_filtered))
    rmse = np.sqrt(np.mean((predictions - speeds_filtered)**2))

    results = {
        'category': category,
        'condition': condition,
        'n_videos': len(speeds_filtered),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'speed_mean': speeds_filtered.mean(),
        'speed_std': speeds_filtered.std(),
        'speeds': speeds_filtered.tolist(),
        'predictions': predictions.tolist()
    }

    logger.info(f"    r={pearson_r:.3f} (p={pearson_p:.2e})")

    return results

def run_transfer_study(steering_vector_path=None, output_dir=None, test_conditions=True, model='vjepa'):
    if steering_vector_path is None:
        steering_vector_path = RESULTS_DIR / f"steering_vector_actual_speed_{model}.pkl"
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading steering vector from {steering_vector_path}")
    with open(steering_vector_path, 'rb') as f:
        steering_data = pickle.load(f)

    steering_vector = steering_data['steering_vector_normalized']
    predictor = steering_data['predictor']
    layer = steering_data['layer']

    logger.info(f"Loaded layer {layer}, training r={steering_data['training_correlation']:.3f}")

    category_results = []
    condition_results = []

    for category in TEST_CATEGORIES:
        logger.info(f"\nCategory: {category.upper()}")

        result = test_category_transfer(steering_vector, predictor, category, model, layer)
        category_results.append(result)

        if test_conditions and category in OBJECT_PROPERTIES:
            logger.info(f"  Testing conditions within {category}...")
            for condition in OBJECT_PROPERTIES[category]:
                cond_result = test_condition_transfer(steering_vector, predictor, category, condition, model, layer)
                if cond_result:
                    condition_results.append(cond_result)

    category_df = pd.DataFrame([
        {
            'category': r['category'],
            'n_videos': r['n_videos'],
            'pearson_r': r['pearson_r'],
            'pearson_p': r['pearson_p'],
            'spearman_r': r['spearman_r'],
            'spearman_p': r['spearman_p'],
            'mae': r['mae'],
            'rmse': r['rmse']
        }
        for r in category_results
    ])
    category_path = output_dir / f"{model}_{CATEGORY_SUMMARIES_FILE}"
    category_df.to_csv(category_path, index=False)
    logger.info(f"Saved category results: {category_path}")

    detailed_path = output_dir / f"{model}_{TRANSFER_RESULTS_FILE}"
    all_results = category_results + condition_results

    def convert_to_python(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = json.loads(json.dumps(all_results, default=convert_to_python))

    with open(detailed_path.with_suffix('.json'), 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved detailed results: {detailed_path.with_suffix('.json')}")

    logger.info("\nSummary:")
    for _, row in category_df.iterrows():
        significance = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
        logger.info(f"{row['category']:10s}: r={row['pearson_r']:+.3f} {significance:3s} (p={row['pearson_p']:.2e}, n={row['n_videos']:3d})")

    return {
        'category_results': category_results,
        'condition_results': condition_results,
        'category_summary': category_df
    }

def main():
    parser = argparse.ArgumentParser(description="Test steering vector transfer across object categories")
    parser.add_argument("--steering_vector", type=str, help="Path to steering vector pickle file")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--test_conditions", action="store_true", default=True, help="Test individual conditions within categories")
    parser.add_argument("--model", type=str, default='vjepa', choices=['vjepa', 'videomae'], help="Model to test")
    args = parser.parse_args()

    run_transfer_study(
        steering_vector_path=args.steering_vector,
        output_dir=args.output_dir,
        test_conditions=args.test_conditions,
        model=args.model
    )


if __name__ == "__main__":
    main()
