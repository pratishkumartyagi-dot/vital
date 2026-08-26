import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge

def run_loso_vitals_validation(feature_matrix, true_spo2, participant_groups):
    """
    Executes Leave-One-Subject-Out (LOSO) clinical cross-validation.
    Ensures zero data leakage from individual frames during training loops.
    """
    logo = LeaveOneGroupOut()
    model_estimator = Ridge(alpha=1.0)
    
    compiled_predictions = []
    compiled_ground_truths = []

    for train_indices, test_indices in logo.split(feature_matrix, true_spo2, groups=participant_groups):
        # Explicit evaluation loop isolation
        X_train, X_test = feature_matrix[train_indices], feature_matrix[test_indices]
        y_train, y_test = true_spo2[train_indices], true_spo2[test_indices]
        
        model_estimator.fit(X_train, y_train)
        predictions = model_estimator.predict(X_test)
        
        compiled_predictions.extend(predictions)
        compiled_ground_truths.extend(y_test)
        
    return np.array(compiled_ground_truths), np.array(compiled_predictions)