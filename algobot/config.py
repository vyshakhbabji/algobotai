"""
AGGRESSIVE PRODUCTION TRADING CONFIGURATION
High-performance settings for meaningful returns
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelConfig:
    """AGGRESSIVE ML model - Lower thresholds for more signals"""
    min_directional_accuracy: float = 0.45         # 45% accuracy (much lower)
    buy_threshold: float = 0.003                   # 0.3% buy threshold (very low)
    sell_threshold: float = -0.003                 # -0.3% sell threshold
    confidence_threshold: float = 0.20             # 20% confidence (very low)
    prediction_window: int = 3                     # 3-day prediction (shorter)
    model_retrain_frequency: int = 20              # Retrain every 20 days
    validation_score_min: float = 0.45             # 45% validation minimum

@dataclass  
class RiskConfig:
    """AGGRESSIVE risk management - Higher positions for better returns"""
    max_position_pct: float = 0.40                # 40% maximum per position (huge increase)
    max_portfolio_risk_pct: float = 0.95          # 95% maximum portfolio risk
    stop_loss_pct: float = 0.12                   # 12% stop loss (wider)
    take_profit_pct: float = 0.25                 # 25% take profit (higher targets)
    trailing_stop_pct: float = 0.08               # 8% trailing stop
    max_daily_loss_pct: float = 0.05              # 5% maximum daily loss
    correlation_limit: float = 0.80               # 80% correlation limit
    min_holding_days: int = 1                     # 1 day minimum hold (very short)

@dataclass
class ExecutionConfig:
    """AGGRESSIVE execution - Daily trading"""
    rebalance_weekdays: Tuple[int, ...] = (0, 1, 2, 3, 4)  # Trade every day
    min_holding_days: int = 1                     # 1 day minimum
    max_holding_days: int = 25                    # 25 day maximum (shorter)
    market_hours_only: bool = True                # Market hours only
    slippage_assumption: float = 0.003            # 0.3% slippage
    min_trade_size: float = 50                    # $50 minimum trade (lower)
    target_positions: int = 12                    # Target 12 positions (more diversification)
    cash_reserve_pct: float = 0.05               # 5% cash reserve (invest more)

@dataclass
class InstitutionalConfig:
    """AGGRESSIVE institutional strategy - Very low thresholds"""
    tier_probs: Tuple[float, ...] = (0.25, 0.32, 0.38, 0.45)  # Much lower tiers!
    breakout_prob: float = 0.25                   # 25% breakout (very low)
    hard_exit_prob: float = 0.15                  # 15% hard exit
    soft_exit_prob: float = 0.20                  # 20% soft exit
    risk_per_trade_pct: float = 0.15              # 15% risk per trade (much higher!)
    early_core_fraction: float = 0.80             # 80% initial position (aggressive)

@dataclass
class UniverseConfig:
    """Stock universe configuration"""
    core_universe: Tuple[str, ...] = (
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX',
        'AMD', 'CRM', 'PLTR', 'SNOW', 'COIN', 'UBER', 'DIS', 'JPM', 
        'BAC', 'JNJ', 'PG', 'KO', 'WMT'
    )
    max_universe: int = 15

@dataclass
class GlobalConfig:
    """Complete global configuration"""
    model: ModelConfig
    risk: RiskConfig
    execution: ExecutionConfig
    institutional: InstitutionalConfig
    universe: UniverseConfig
    production_mode: bool = True
    optimization_level: str = 'AGGRESSIVE'
    strategy_type: str = 'HIGH_PERFORMANCE_ML_INSTITUTIONAL'

# Global configuration instance - AGGRESSIVE PRODUCTION MODE
GLOBAL_CONFIG = GlobalConfig(
    model=ModelConfig(),
    risk=RiskConfig(), 
    execution=ExecutionConfig(),
    institutional=InstitutionalConfig(),
    universe=UniverseConfig()
)

print("🚀 AGGRESSIVE PRODUCTION CONFIGURATION LOADED")
print("   • Ultra-low ML thresholds (45% accuracy, 20% confidence)")
print("   • High-risk management (40% max position, 15% risk per trade)")
print("   • Daily rebalancing for maximum opportunity capture")
print("   • Target: 20-40% annual returns")
