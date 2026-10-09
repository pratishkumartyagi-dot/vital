"""
CatBoost Models Module
=======================

Implements CatBoost regression and classification models for:
- Blood Pressure (SBP, DBP) estimation - Regression
- Blood Glucose (HbA1c) estimation - Regression  
- Mental Stress & Cognitive Workload classification - Classification
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: catboost not installed. Install with: pip install catboost")


class BaseCatBoostModel:
    """Base class for CatBoost models."""
    
    def __init__(self, model_dir: Optional[str] = None):
        if not CATBOOST_AVAILABLE:
            raise ImportError("catboost is required. Install via: pip install catboost")
        self.model_dir = Path(model_dir) if model_dir else Path.cwd()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.metrics = {}
    
    def save_model(self, filename: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained")
        model_path = self.model_dir / filename
        self.model.save_model(str(model_path))
        
        # Save metadata
        meta_path = model_path.with_suffix('.meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, filename: str) -> None:
        """Load trained model."""
        model_path = self.model_dir / filename
        meta_path = model_path.with_suffix('.meta.pkl')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = self._load_catboost_model(str(model_path))
        
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.feature_names = meta.get('feature_names', [])
                self.metrics = meta.get('metrics', {})
                self.is_trained = meta.get('is_trained', True)
    
    def _load_catboost_model(self, path: str):
        raise NotImplementedError
    
    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction/training."""
        if len(self.feature_names) > 0:
            # Use only trained features
            missing = [f for f in self.feature_names if f not in features_df.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            return features_df[self.feature_names]
        return features_df


class CatBoostBPModel(BaseCatBoostModel):
    """
    CatBoost model for cuffless Blood Pressure estimation (SBP, DBP).
    
    Based on research showing CatBoost achieves MAE ~2.38 mmHg for SBP
    and ~1.52 mmHg for DBP using PPG features.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        super().__init__(model_dir)
        self.sbp_model = None
        self.dbp_model = None
    
    def _load_catboost_model(self, path: str):
        return CatBoostRegressor().load_model(path)
    
    def train(self, 
              features_df: pd.DataFrame,
              sbp_targets: Union[np.ndarray, pd.Series],
              dbp_targets: Union[np.ndarray, pd.Series],
              params: Optional[Dict[str, Any]] = None,
              validation_split: float = 0.2,
              **kwargs) -> Dict[str, Any]:
        """
        Train BP prediction models for SBP and DBP.
        
        Args:
            features_df: DataFrame of extracted PPG features
            sbp_targets: Systolic blood pressure targets (mmHg)
            dbp_targets: Diastolic blood pressure targets (mmHg)
            params: CatBoost parameters
        """
        default_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'random_seed': 42,
            'early_stopping_rounds': 100,
            'verbose': 500,
            'use_best_model': True
        }
        if params:
            default_params.update(params)
        
        # Convert targets to numpy
        sbp = np.array(sbp_targets, dtype=float)
        dbp = np.array(dbp_targets, dtype=float)
        
        # Split validation
        n = len(features_df)
        val_size = int(n * validation_split)
        idx = np.random.permutation(n)
        train_idx = idx[val_size:]
        val_idx = idx[:val_size]
        
        X_train = features_df.iloc[train_idx]
        X_val = features_df.iloc[val_idx]
        y_sbp_train = sbp[train_idx]
        y_sbp_val = sbp[val_idx]
        y_dbp_train = dbp[train_idx]
        y_dbp_val = dbp[val_idx]
        
        self.feature_names = list(features_df.columns)
        
        # Train SBP model
        train_pool = Pool(X_train, y_sbp_train)
        val_pool = Pool(X_val, y_sbp_val)
        
        self.sbp_model = CatBoostRegressor(**default_params)
        sbp_metrics = self.sbp_model.fit(train_pool, eval_set=val_pool)
        
        # Train DBP model
        train_pool_dbp = Pool(X_train, y_dbp_train)
        val_pool_dbp = Pool(X_val, y_dbp_val)
        
        self.dbp_model = CatBoostRegressor(**default_params)
        dbp_metrics = self.dbp_model.fit(train_pool_dbp, eval_set=val_pool_dbp)
        
        self.is_trained = True
        self.metrics = {
            'sbp_best_score': self.sbp_model.best_score_,
            'dbp_best_score': self.dbp_model.best_score_,
            'sbp_iterations': self.sbp_model.tree_count_
        }
        
        return self.metrics
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict SBP and DBP.
        
        Returns:
            Tuple of (sbp_predictions, dbp_predictions) in mmHg
        """
        if self.sbp_model is None or self.dbp_model is None:
            raise ValueError("Models not trained/loaded")
        
        X = self._prepare_features(features_df)
        
        sbp_pred = self.sbp_model.predict(X)
        dbp_pred = self.dbp_model.predict(X)
        
        # Clamp to physiologically reasonable ranges
        sbp_pred = np.clip(sbp_pred, 70, 200)
        dbp_pred = np.clip(dbp_pred, 40, 130)
        
        return sbp_pred.astype(float), dbp_pred.astype(float)
    
    def save_model(self, filename: str = 'catboost_bp.cbm') -> None:
        """Save both SBP and DBP models."""
        if self.sbp_model is None or self.dbp_model is None:
            raise ValueError("Models not trained")
        
        # Save SBP
        sbp_path = self.model_dir / f'sbp_{filename}'
        self.sbp_model.save_model(str(sbp_path))
        
        # Save DBP
        dbp_path = self.model_dir / f'dbp_{filename}'
        self.dbp_model.save_model(str(dbp_path))
        
        # Save metadata
        meta_path = self.model_dir / f'bp_{filename}.meta.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'is_trained': self.is_trained
            }, f)
    
    def load_model(self, filename: str = 'catboost_bp.cbm') -> None:
        """Load both SBP and DBP models."""
        meta_path = self.model_dir / f'bp_{filename}.meta.pkl'
        sbp_path = self.model_dir / f'sbp_{filename}'
        dbp_path = self.model_dir / f'dbp_{filename}'
        
        if not sbp_path.exists() or not dbp_path.exists():
            raise FileNotFoundError("BP models not found")
        
        self.sbp_model = CatBoostRegressor().load_model(str(sbp_path))
        self.dbp_model = CatBoostRegressor().load_model(str(dbp_path))
        
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                self.feature_names = meta.get('feature_names', [])
                self.metrics = meta.get('metrics', {})
                self.is_trained = meta.get('is_trained', True)


class CatBoostGlucoseModel(BaseCatBoostModel):
    """
    CatBoost model for non-invasive blood glucose (HbA1c) estimation.
    
    Research shows CatBoost achieves R² score of 0.92 for glucose prediction.
    """
    
    def _load_catboost_model(self, path: str):
        return CatBoostRegressor().load_model(path)
    
    def train(self,
              features_df: pd.DataFrame,
              glucose_targets: Union[np.ndarray, pd.Series],
              params: Optional[Dict[str, Any]] = None,
              validation_split: float = 0.2,
              **kwargs) -> Dict[str, Any]:
        """Train glucose prediction model."""
        default_params = {
            'iterations': 1500,
            'learning_rate': 0.02,
            'depth': 7,
            'l2_leaf_reg': 4,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'early_stopping_rounds': 80,
            'verbose': 500
        }
        if params:
            default_params.update(params)
        
        glucose = np.array(glucose_targets, dtype=float)
        
        n = len(features_df)
        val_size = int(n * validation_split)
        idx = np.random.permutation(n)
        train_idx = idx[val_size:]
        val_idx = idx[:val_size]
        
        X_train = features_df.iloc[train_idx]
        X_val = features_df.iloc[val_idx]
        y_train = glucose[train_idx]
        y_val = glucose[val_idx]
        
        self.feature_names = list(features_df.columns)
        
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        self.model = CatBoostRegressor(**default_params)
        self.model.fit(train_pool, eval_set=val_pool)
        
        self.is_trained = True
        self.metrics = {
            'best_score': self.model.best_score_,
            'iterations': self.model.tree_count_
        }
        return self.metrics
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict blood glucose levels."""
        if self.model is None:
            raise ValueError("Model not trained/loaded")
        
        X = self._prepare_features(features_df)
        pred = self.model.predict(X)
        # Clamp to clinically reasonable ranges (mg/dL)
        pred = np.clip(pred, 60, 250)
        return pred.astype(float)


class CatBoostStressModel(BaseCatBoostModel):
    """
    CatBoost model for mental stress and cognitive workload classification.
    
    Classifies stress levels based on PPG-derived HRV and morphological features.
    Achieves ~84% accuracy as reported in literature.
    """
    
    def __init__(self, model_dir: Optional[str] = None, num_classes: int = 3):
        super().__init__(model_dir)
        self.num_classes = num_classes
        self.class_labels = ["low", "medium", "high"] if num_classes == 3 else [str(i) for i in range(num_classes)]
    
    def _load_catboost_model(self, path: str):
        return CatBoostClassifier().load_model(path)
    
    def train(self,
              features_df: pd.DataFrame,
              stress_labels: Union[np.ndarray, List],
              params: Optional[Dict[str, Any]] = None,
              validation_split: float = 0.2,
              **kwargs) -> Dict[str, Any]:
        """Train stress classification model."""
        default_params = {
            'iterations': 1200,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': 42,
            'early_stopping_rounds': 60,
            'verbose': 500,
            'auto_class_weights': 'Balanced'
        }
        if params:
            default_params.update(params)
        
        labels = np.array(stress_labels)
        
        n = len(features_df)
        val_size = int(n * validation_split)
        idx = np.random.permutation(n)
        train_idx = idx[val_size:]
        val_idx = idx[:val_size]
        
        X_train = features_df.iloc[train_idx]
        X_val = features_df.iloc[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        
        self.feature_names = list(features_df.columns)
        
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        self.model = CatBoostClassifier(**default_params)
        self.model.fit(train_pool, eval_set=val_pool)
        
        self.is_trained = True
        self.metrics = {
            'best_score': self.model.best_score_,
            'iterations': self.model.tree_count_
        }
        return self.metrics
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict stress levels.
        
        Returns:
            Tuple of (class_predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained/loaded")
        
        X = self._prepare_features(features_df)
        probs = self.model.predict_proba(X)
        preds = self.model.predict(X)
        return preds, probs
    
    def predict_stress_level(self, features_df: pd.DataFrame) -> List[str]:
        """Predict stress level as strings."""
        preds, _ = self.predict(features_df)
        return [self.class_labels[int(p)] if p in range(self.num_classes) else 'unknown' for p in preds]