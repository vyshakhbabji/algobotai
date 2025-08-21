"""Basic, robust feature engineering."""
import pandas as pd

FEATURE_COLUMNS = [
    'returns', 'price_vs_ma5', 'price_vs_ma20', 'momentum_5', 'momentum_20', 'volume_ratio'
]


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # Defensive extraction of scalar series
    def _s(col):
        s = data[col]
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0]
        return s
    close = _s('Close').astype(float)
    volume = _s('Volume').astype(float) if 'Volume' in data.columns else None
    data['returns'] = close.pct_change()
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    data['ma_5'] = ma5
    data['ma_20'] = ma20
    data['price_vs_ma5'] = (close / ma5) - 1
    data['price_vs_ma20'] = (close / ma20) - 1
    data['momentum_5'] = (close / close.shift(5)) - 1
    data['momentum_20'] = (close / close.shift(20)) - 1
    if volume is not None:
        vol_ma20 = volume.rolling(20).mean()
        data['volume_ma_20'] = vol_ma20
        data['volume_ratio'] = (volume / vol_ma20) - 1
    else:
        data['volume_ma_20'] = 0.0
        data['volume_ratio'] = 0.0
    data['target'] = data['returns'].shift(-1)
    data = data.dropna()
    return data
