import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold

def compute_baselines(activations, values, n_random=100):
    baselines = {}

    random_scores = []
    for seed in range(n_random):
        np.random.seed(seed)
        random_vector = np.random.randn(activations.shape[1])
        random_vector /= np.linalg.norm(random_vector)
        random_proj = activations @ random_vector
        r, _ = pearsonr(random_proj, values)
        random_scores.append(abs(r))

    baselines['random_mean'] = np.mean(random_scores)
    baselines['random_std'] = np.std(random_scores)
    baselines['random_max'] = np.max(random_scores)
    baselines['random_scores'] = random_scores

    mean_act = activations.mean(axis=0)
    mean_act_normalized = mean_act / np.linalg.norm(mean_act)
    mean_proj = activations @ mean_act_normalized
    r_mean, _ = pearsonr(mean_proj, values)
    baselines['mean_activation'] = abs(r_mean)

    pca = PCA(n_components=min(10, activations.shape[0]))
    pca_projs = pca.fit_transform(activations)

    pca_scores = []
    for i in range(min(5, pca_projs.shape[1])):
        r, _ = pearsonr(pca_projs[:, i], values)
        pca_scores.append(abs(r))

    baselines['pca_pc1'] = pca_scores[0] if len(pca_scores) > 0 else 0
    baselines['pca_pc2'] = pca_scores[1] if len(pca_scores) > 1 else 0
    baselines['pca_pc3'] = pca_scores[2] if len(pca_scores) > 2 else 0
    baselines['pca_pc4'] = pca_scores[3] if len(pca_scores) > 3 else 0
    baselines['pca_pc5'] = pca_scores[4] if len(pca_scores) > 4 else 0
    baselines['pca_var_explained'] = pca.explained_variance_ratio_[0]

    ridge = Ridge(alpha=1.0)
    ridge.fit(activations, values)
    ridge_pred = ridge.predict(activations)
    r_ridge, _ = pearsonr(ridge_pred, values)
    baselines['ridge'] = abs(r_ridge)

    linear = LinearRegression()
    linear.fit(activations, values)
    linear_pred = linear.predict(activations)
    r_linear, _ = pearsonr(linear_pred, values)
    baselines['linear'] = abs(r_linear)

    return baselines

def cross_validate_steering(activations, values, steering_method, n_splits=5, **method_kwargs):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kfold.split(activations):
        train_acts = activations[train_idx]
        train_vals = values[train_idx]

        if 'df' in method_kwargs:
            train_df = method_kwargs['df'].iloc[train_idx].reset_index(drop=True)
            method_kwargs_train = {**method_kwargs, 'df': train_df}
        else:
            method_kwargs_train = method_kwargs

        steering_info = steering_method(train_acts, train_vals, **method_kwargs_train)
        steering = steering_info['steering_vector_normalized']

        val_acts = activations[val_idx]
        val_vals = values[val_idx]

        val_proj = val_acts @ steering
        r, _ = pearsonr(val_proj, val_vals)
        cv_scores.append(abs(r))

    return {
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'cv_scores': cv_scores
    }
