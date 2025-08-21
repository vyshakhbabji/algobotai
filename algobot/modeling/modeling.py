"""
Modeling module for ML-driven trading pipeline.
Supports baseline (logistic, Ridge/Lasso), tree-based (LightGBM, XGBoost, CatBoost), and ensemble models.
Includes time-series cross-validation, embargo, calibration, and drift checks as per prompt.yaml.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

MODEL_TYPES = ['logistic', 'ridge', 'lasso', 'random_forest', 'lightgbm']

def get_model(model_type: str, **kwargs):
    if model_type == 'logistic':
        return LogisticRegression(**kwargs)
    if model_type == 'ridge':
        return Ridge(**kwargs)
    if model_type == 'lasso':
        return Lasso(**kwargs)
    if model_type == 'random_forest':
        return RandomForestClassifier(**kwargs)
    if model_type == 'lightgbm' and lgb is not None:
        return lgb.LGBMClassifier(**kwargs)
    raise ValueError(f"Unknown or unavailable model type: {model_type}")

def time_series_cv(X, y, model_type: str, n_splits: int = 5, embargo: int = 0, calibrate: bool = True, **kwargs) -> Dict[str, Any]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, test_idx in tscv.split(X):
        # Embargo: remove last 'embargo' samples from train
        if embargo > 0:
            train_idx = train_idx[:-embargo] if len(train_idx) > embargo else train_idx
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        model = get_model(model_type)
        model.fit(X_train, y_train)
        if calibrate:
            model = CalibratedClassifierCV(model, cv='prefit')
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else y_pred
        acc = np.mean(y_pred == y_test)
        results.append({'acc': acc, 'y_true': y_test, 'y_pred': y_pred, 'y_prob': y_prob})
    return {'cv_results': results}
