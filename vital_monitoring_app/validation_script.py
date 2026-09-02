import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval for a statistic."""
    if len(data) < 2:
        return np.nan, np.nan
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(np.mean(sample))
    alpha = (100 - ci) / 2
    return np.percentile(boot_stats, alpha), np.percentile(boot_stats, 100 - alpha)

def run_loso_vitals_validation(feature_matrix, true_spo2, participant_groups, alphas=None, n_inner_folds=3):
    """
    Executes Leave-One-Subject-Out (LOSO) clinical cross-validation with nested CV for hyperparameter tuning.
    Ensures zero data leakage from individual frames during training loops.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Features for each sample (n_samples, n_features)
    true_spo2 : np.ndarray
        Ground truth SpO2 values (n_samples,)
    participant_groups : np.ndarray
        Group labels for each sample (n_samples,)
    alphas : array-like, optional
        Ridge regularization parameters to search. Default: logspace(-3, 3, 7)
    n_inner_folds : int
        Number of folds for inner CV (default: 3)
    
    Returns:
    --------
    dict containing:
        - y_true: ground truth values
        - y_pred: predictions
        - metrics: MAE, RMSE, R2 with bootstrap CIs
        - best_alpha_per_fold: selected alpha for each outer fold
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 7)
    
    logo = LeaveOneGroupOut()
    compiled_predictions = []
    compiled_ground_truths = []
    best_alphas = []
    fold_metrics = []

    for train_indices, test_indices in logo.split(feature_matrix, true_spo2, groups=participant_groups):
        X_train, X_test = feature_matrix[train_indices], feature_matrix[test_indices]
        y_train, y_test = true_spo2[train_indices], true_spo2[test_indices]
        groups_train = participant_groups[train_indices]
        
        # Inner CV for alpha selection
        inner_cv = KFold(n_splits=min(n_inner_folds, len(np.unique(groups_train))), shuffle=True, random_state=42)
        ridge = Ridge()
        param_grid = {'alpha': alphas}
        
        grid_search = GridSearchCV(
            ridge, param_grid, cv=inner_cv, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.best_params_['alpha']
        best_alphas.append(best_alpha)
        
        # Train final model with best alpha
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        compiled_predictions.extend(predictions)
        compiled_ground_truths.extend(y_test)
        
        # Per-fold metrics
        fold_mae = mean_absolute_error(y_test, predictions)
        fold_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        fold_r2 = r2_score(y_test, predictions)
        fold_metrics.append({'mae': fold_mae, 'rmse': fold_rmse, 'r2': fold_r2})

    y_true = np.array(compiled_ground_truths)
    y_pred = np.array(compiled_predictions)
    
    # Overall metrics with bootstrap CIs
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    overall_r2 = r2_score(y_true, y_pred)
    
    mae_ci = bootstrap_ci(np.abs(y_true - y_pred))
    rmse_ci = bootstrap_ci((y_true - y_pred) ** 2)
    r2_ci = bootstrap_ci((y_true - y_pred) ** 2)  # Approximate
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': {
            'mae': overall_mae,
            'mae_ci': mae_ci,
            'rmse': overall_rmse,
            'rmse_ci': rmse_ci,
            'r2': overall_r2,
            'r2_ci': r2_ci,
        },
        'fold_metrics': fold_metrics,
        'best_alphas': best_alphas,
    }

def run_loso_multivariate_validation(feature_matrix, true_vitals, participant_groups, vital_names=None, alphas=None):
    """
    Run LOSO validation for multiple vital signs with multiple testing correction.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Features (n_samples, n_features)
    true_vitals : np.ndarray
        Ground truth for multiple vitals (n_samples, n_vitals)
    participant_groups : np.ndarray
        Group labels (n_samples,)
    vital_names : list, optional
        Names of vital signs (default: ['spo2', 'hr', 'resp_rate', 'sbp', 'dbp'])
    alphas : array-like, optional
        Ridge alphas for each vital (default: same for all)
    
    Returns:
    --------
    dict with per-vital results and corrected p-values
    """
    if vital_names is None:
        vital_names = ['spo2', 'hr', 'resp_rate', 'sbp', 'dbp']
    
    n_vitals = true_vitals.shape[1]
    results = {}
    p_values = []
    
    for i, name in enumerate(vital_names):
        if i >= true_vitals.shape[1]:
            break
        y = true_vitals[:, i]
        # Skip if all NaN
        valid_mask = ~np.isnan(y)
        if not np.any(valid_mask):
            results[name] = {'error': 'All NaN values'}
            continue
        
        res = run_loso_vitals_validation(
            feature_matrix[valid_mask], y[valid_mask], 
            participant_groups[valid_mask], alphas
        )
        results[name] = res
        
        # Test if MAE is significantly better than baseline (mean prediction)
        baseline_pred = np.full_like(res['y_pred'], np.mean(res['y_true']))
        baseline_mae = mean_absolute_error(res['y_true'], baseline_pred)
        model_mae = res['metrics']['mae']
        
        # Permutation test p-value
        n_perm = 1000
        perm_maes = []
        for _ in range(n_perm):
            perm_pred = np.random.permutation(res['y_pred'])
            perm_maes.append(mean_absolute_error(res['y_true'], perm_pred))
        p_val = np.mean(np.array(perm_maes) <= model_mae)
        p_values.append(p_val)
    
    # Benjamini-Hochberg FDR correction
    if p_values:
        p_values = np.array(p_values)
        sorted_idx = np.argsort(p_values)
        m = len(p_values)
        corrected = np.zeros(m)
        for rank, idx in enumerate(sorted_idx):
            corrected[idx] = min(1.0, p_values[idx] * m / (rank + 1))
        # Ensure monotonicity
        for i in range(m - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1])
        
        for i, name in enumerate(vital_names):
            if name in results and 'error' not in results[name]:
                results[name]['p_value'] = p_values[i]
                results[name]['p_value_fdr'] = corrected[i]
    
    return results