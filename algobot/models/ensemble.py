"""Lightweight ensemble training (Ridge + RandomForest) for next-day return prediction.

Replaces legacy EliteAITrader adapter with a transparent, testable utility.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


@dataclass
class TrainedEnsemble:
    models: Dict[str, object]
    feature_cols: List[str]
    weights: Dict[str, float]
    train_pred: pd.Series  # blended in-sample predictions
    symbol: str

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for name, m in self.models.items():
            try:
                preds.append(self.weights[name] * m.predict(X[self.feature_cols]))
            except Exception:
                preds.append(self.weights[name] * m.predict(X))
        return np.sum(preds, axis=0)


def train_ensemble(train_df: pd.DataFrame, feature_cols: List[str], target_col: str = 'target',
                   k: int = 3, random_state: int = 42) -> TrainedEnsemble:
    if train_df.empty:
        raise ValueError("Empty training data for ensemble")
    X = train_df[feature_cols]
    y = train_df[target_col]

    models: Dict[str, object] = {
        'ridge': Ridge(alpha=1.0),
        'rf': RandomForestRegressor(n_estimators=160, max_depth=6, random_state=random_state, n_jobs=-1)
    }

    # Determine folds (avoid too many folds for tiny data)
    n_splits = min(k, max(2, len(train_df)//60)) if len(train_df) >= 120 else 2
    kf = TimeSeriesSplit(n_splits=n_splits)
    cv_scores: Dict[str, float] = {}
    for name, model in models.items():
        fold_errs = []
        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            fold_errs.append(mean_squared_error(yva, pred))
        cv_scores[name] = float(np.mean(fold_errs)) if fold_errs else 1.0

    inv = {n: 1.0 / max(s, 1e-9) for n, s in cv_scores.items()}
    denom = sum(inv.values())
    weights = {n: v / denom for n, v in inv.items()}

    # Refit full
    for name, model in models.items():
        model.fit(X, y)

    blended = sum(weights[n] * models[n].predict(X) for n in models)
    train_pred = pd.Series(blended, index=train_df.index, name='blended_pred')

    return TrainedEnsemble(models=models, feature_cols=feature_cols, weights=weights, train_pred=train_pred, symbol=str(getattr(train_df.index, 'name', 'symbol')))
