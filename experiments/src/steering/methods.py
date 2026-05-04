import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def compute_steering_vector_percentile(activations, values, low_percentile=33, high_percentile=67):
    low_thresh = np.percentile(values, low_percentile)
    high_thresh = np.percentile(values, high_percentile)

    low_mask = values <= low_thresh
    high_mask = values >= high_thresh

    low_acts = activations[low_mask]
    high_acts = activations[high_mask]

    low_mean = low_acts.mean(axis=0)
    high_mean = high_acts.mean(axis=0)

    steering = high_mean - low_mean
    steering_normalized = steering / np.linalg.norm(steering)

    projection = activations @ steering_normalized
    r, p = pearsonr(projection, values)

    effect_size = (high_mean - low_mean).mean() / activations.std()

    return {
        'steering_vector': steering,
        'steering_vector_normalized': steering_normalized,
        'low_mean': low_mean,
        'high_mean': high_mean,
        'low_count': low_mask.sum(),
        'high_count': high_mask.sum(),
        'low_value_mean': values[low_mask].mean(),
        'high_value_mean': values[high_mask].mean(),
        'correlation': r,
        'p_value': p,
        'effect_size': effect_size
    }

def compute_steering_vector_stratified(activations, values, df, strata_columns, n_per_stratum=2):
    low_indices = []
    high_indices = []

    unique_strata = df[strata_columns].drop_duplicates()

    for _, stratum in unique_strata.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for col in strata_columns:
            mask &= (df[col] == stratum[col])

        stratum_indices = np.where(mask)[0]
        stratum_values = values[stratum_indices]

        sorted_idx = stratum_indices[np.argsort(stratum_values)]

        n_available = len(sorted_idx)
        n_take = min(n_per_stratum, n_available // 2)

        if n_take > 0:
            low_indices.extend(sorted_idx[:n_take])
            high_indices.extend(sorted_idx[-n_take:])

    low_indices = np.array(low_indices)
    high_indices = np.array(high_indices)

    low_acts = activations[low_indices]
    high_acts = activations[high_indices]

    low_mean = low_acts.mean(axis=0)
    high_mean = high_acts.mean(axis=0)

    steering = high_mean - low_mean
    steering_normalized = steering / np.linalg.norm(steering)

    projection = activations @ steering_normalized
    r, p = pearsonr(projection, values)

    effect_size = (high_mean - low_mean).mean() / activations.std()

    low_strata_counts = df.iloc[low_indices][strata_columns].value_counts()
    high_strata_counts = df.iloc[high_indices][strata_columns].value_counts()

    return {
        'steering_vector': steering,
        'steering_vector_normalized': steering_normalized,
        'low_mean': low_mean,
        'high_mean': high_mean,
        'low_count': len(low_indices),
        'high_count': len(high_indices),
        'low_value_mean': values[low_indices].mean(),
        'high_value_mean': values[high_indices].mean(),
        'correlation': r,
        'p_value': p,
        'effect_size': effect_size,
        'low_indices': low_indices,
        'high_indices': high_indices,
        'low_strata_distribution': low_strata_counts.to_dict(),
        'high_strata_distribution': high_strata_counts.to_dict()
    }

def train_predictor(activations, steering_vector, values):
    projections = activations @ steering_vector

    model = LinearRegression()
    model.fit(projections.reshape(-1, 1), values)

    predictions = model.predict(projections.reshape(-1, 1))

    r, _ = pearsonr(predictions, values)
    r2 = r ** 2

    return model, predictions, projections, r, r2
