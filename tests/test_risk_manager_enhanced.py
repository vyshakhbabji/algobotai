import pandas as pd
import numpy as np
from algobot.risk.manager import RiskManagerCore, TradeLimits

def test_atr_and_var():
    rm = RiskManagerCore(TradeLimits())
    # create price data
    dates = pd.date_range('2024-01-01', periods=50, freq='B')
    close = np.linspace(100,110,50) + np.random.normal(0,1,50)
    high = close + np.random.uniform(0.5,1.5,50)
    low = close - np.random.uniform(0.5,1.5,50)
    df = pd.DataFrame({'High':high,'Low':low,'Close':close}, index=dates)
    atr_val = rm.atr(df)
    assert atr_val is not None
    stop = rm.atr_stop_loss(entry=105, side='BUY', atr_value=atr_val, multiple=2)
    assert stop < 105
    # feed returns
    for r in np.random.normal(0,0.01,60):
        rm.update_var_history(r)
    var95 = rm.value_at_risk(0.95)
    assert var95 is not None
