import pandas as pd
import numpy as np
from algobot.portfolio import portfolio_scanner as ps


def test_ml_signal_integration(monkeypatch):
    # mock download to avoid network
    def fake_download(symbol, start, end, use_local=True):
        idx = pd.date_range('2024-01-01', periods=200)
        close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
        return close

    monkeypatch.setattr(ps, '_download', fake_download)
    monkeypatch.setattr(ps, '_strength_score', lambda close: 0.0)

    res = ps.scan_universe(['AAPL'], thresholds=(0.5, -0.5), use_ml=True,
                            ml_signals={'AAPL': 1.0}, ml_weight=1.0,
                            use_local=False, write_json=False)
    assert res['results']['AAPL']['classification'] == 'strong'
    assert res['results']['AAPL']['ml_signal'] == 1.0
    assert res['results']['AAPL']['combined_score'] == 1.0
