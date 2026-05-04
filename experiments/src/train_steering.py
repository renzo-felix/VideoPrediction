import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
import argparse
import logging

from config import *
from analysis import load_activations
from steering import (
    compute_steering_vector_percentile,
    compute_steering_vector_stratified,
    train_predictor,
    compute_baselines,
    cross_validate_steering
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_steering_vector(concept='actual_speed', model='vjepa', activations_dir=None,
                         metadata_path=None, layer=VJEPA_LAYER, output_path=None,
                         method='stratified', strata_columns=None, n_per_stratum=2,
                         low_percentile=STEERING_LOW_PERCENTILE, high_percentile=STEERING_HIGH_PERCENTILE):

    if activations_dir is None:
        activations_dir = ACTIVATIONS_DIR / f"training_{model}"
    if metadata_path is None:
        metadata_path = DATA_DIR / TRAINING_METADATA
    if output_path is None:
        suffix = '_stratified' if method == 'stratified' else ''
        output_path = RESULTS_DIR / f"steering_vector_{concept}_{model}{suffix}.pkl"

    logger.info(f"Model: {model}")
    logger.info(f"Concept: {concept}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Method: {method}")
    logger.info(f"Activations: {activations_dir}")

    activations, values, _, df = load_activations(activations_dir, metadata_path, layer)

    logger.info(f"Loaded metadata: {len(df)} videos")
    logger.info(f"Loaded {len(activations)} activations")
    logger.info(f"Speed range: [{values.min():.2f}, {values.max():.2f}]")

    if method == 'stratified':
        if strata_columns is None:
            available_columns = df.columns.tolist()
            strata_columns = [col for col in ['size', 'color', 'shape'] if col in available_columns]

        if not strata_columns:
            logger.warning("No strata columns found, falling back to percentile method")
            method = 'percentile'

    if method == 'stratified':
        logger.info(f"Using stratified method with strata: {strata_columns}, n_per_stratum={n_per_stratum}")
        steering_info = compute_steering_vector_stratified(
            activations, values, df, strata_columns, n_per_stratum
        )

        logger.info(f"\nStratification balance:")
        logger.info(f"  Low group distribution: {steering_info['low_strata_distribution']}")
        logger.info(f"  High group distribution: {steering_info['high_strata_distribution']}")

    else:
        logger.info(f"Using percentile method: {low_percentile}th - {high_percentile}th")
        steering_info = compute_steering_vector_percentile(
            activations, values, low_percentile, high_percentile
        )

    steering = steering_info['steering_vector_normalized']

    logger.info(f"Low {concept}:  n={steering_info['low_count']}, mean={steering_info['low_value_mean']:.2f}")
    logger.info(f"High {concept}: n={steering_info['high_count']}, mean={steering_info['high_value_mean']:.2f}")
    logger.info(f"Cohen's d: {steering_info['effect_size']:.3f}")

    model_pred, predictions, projections, r, r2 = train_predictor(activations, steering, values)

    logger.info(f"Pearson r: {r:.3f}")
    logger.info(f"R²:        {r2:.3f}")

    logger.info("\nBaseline Comparisons:")
    baselines = compute_baselines(activations, values, n_random=100)
    logger.info(f"  Random (n=100):  r={baselines['random_mean']:.3f} ± {baselines['random_std']:.3f} (max={baselines['random_max']:.3f})")
    logger.info(f"  Mean activation: r={baselines['mean_activation']:.3f}")
    logger.info(f"  PCA PC1:         r={baselines['pca_pc1']:.3f} (explains {baselines['pca_var_explained']:.1%} variance)")
    logger.info(f"  PCA PC2:         r={baselines['pca_pc2']:.3f}")
    logger.info(f"  PCA PC3:         r={baselines['pca_pc3']:.3f}")
    logger.info(f"  Ridge (L2):      r={baselines['ridge']:.3f}")
    logger.info(f"  Linear (full):   r={baselines['linear']:.3f}")
    logger.info(f"  Steering:        r={abs(r):.3f} ⭐")

    logger.info("\nCross-validation (5-fold):")
    if method == 'stratified':
        cv_results = cross_validate_steering(
            activations, values, compute_steering_vector_stratified,
            n_splits=5, df=df, strata_columns=strata_columns, n_per_stratum=n_per_stratum
        )
    else:
        cv_results = cross_validate_steering(
            activations, values, compute_steering_vector_percentile,
            n_splits=5, low_percentile=low_percentile, high_percentile=high_percentile
        )

    logger.info(f"  CV mean: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}")
    logger.info(f"  Training: {abs(r):.3f}")
    logger.info(f"  Difference: {abs(abs(r) - cv_results['cv_mean']):.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'steering_vector': steering_info['steering_vector'],
        'steering_vector_normalized': steering,
        'predictor': model_pred,
        'training_correlation': abs(r),
        'training_r2': r2,
        'effect_size': steering_info['effect_size'],
        'baselines': baselines,
        'cross_validation': cv_results,
        'model': model,
        'concept': concept,
        'layer': layer,
        'method': method,
        'metadata_path': str(metadata_path),
        'activations_dir': str(activations_dir)
    }

    if method == 'stratified':
        results['strata_columns'] = strata_columns
        results['n_per_stratum'] = n_per_stratum
        results['low_strata_distribution'] = steering_info['low_strata_distribution']
        results['high_strata_distribution'] = steering_info['high_strata_distribution']
    else:
        results['low_percentile'] = low_percentile
        results['high_percentile'] = high_percentile

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"✓ Saved: {output_path}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vjepa", choices=["vjepa", "videomae"])
    parser.add_argument("--concept", default="actual_speed")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--activations-dir", type=str)
    parser.add_argument("--metadata-path", type=str)
    parser.add_argument("--low-percentile", type=float, default=STEERING_LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=STEERING_HIGH_PERCENTILE)
    parser.add_argument("--output", type=str)
    parser.add_argument("--multi", action="store_true", help="Use multi-attribute training set")
    parser.add_argument("--method", choices=["percentile", "stratified"], default="stratified",
                       help="Steering vector extraction method")
    parser.add_argument("--strata", nargs="+", default=None,
                       help="Columns to stratify by (e.g., size color)")
    parser.add_argument("--n-per-stratum", type=int, default=2,
                       help="Number of samples per stratum for stratified method")

    args = parser.parse_args()

    layer = args.layer if args.layer is not None else (VJEPA_LAYER if args.model == "vjepa" else VIDEOMAE_LAYER)

    activations_dir = None
    metadata_path = None
    output_path = Path(args.output) if args.output else None

    if args.multi:
        activations_dir = ACTIVATIONS_DIR / f"training_multi_{args.model}"
        metadata_path = DATA_DIR / "training_multi_attribute_metadata.csv"
        if output_path is None:
            suffix = '_stratified' if args.method == 'stratified' else ''
            output_path = RESULTS_DIR / f"steering_vector_{args.concept}_{args.model}_multi{suffix}.pkl"
    elif args.activations_dir:
        activations_dir = Path(args.activations_dir)

    if args.metadata_path:
        metadata_path = Path(args.metadata_path)

    results = train_steering_vector(
        model=args.model,
        concept=args.concept,
        layer=layer,
        activations_dir=activations_dir,
        metadata_path=metadata_path,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        output_path=output_path,
        method=args.method,
        strata_columns=args.strata,
        n_per_stratum=args.n_per_stratum
    )

    print(f"{args.model}/{args.concept}: r={results['training_correlation']:.3f}, R²={results['training_r2']:.3f}, d={results['effect_size']:.3f}")

if __name__ == "__main__":
    main()
