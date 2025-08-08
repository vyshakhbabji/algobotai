#!/usr/bin/env python3
"""
Background AI Model Monitor and Auto-Optimizer
Runs continuously to monitor model performance and optimize when needed
"""

import os
import sys
import time
import json
import threading
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pages'))

try:
    from pages.ai_optimizer import SelfImprovingAI
except ImportError:
    # Alternative import path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from ai_optimizer import SelfImprovingAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_monitor.log'),
        logging.StreamHandler()
    ]
)

class AIModelMonitor:
    def __init__(self):
        self.ai_optimizer = SelfImprovingAI()
        self.monitoring_enabled = True
        self.optimization_threshold = 0.5  # Optimize if performance below 50%
        self.evaluation_interval_hours = 6  # Evaluate every 6 hours
        self.min_optimization_interval_hours = 24  # Don't optimize more than once per day per stock
        
        # Load monitoring config
        self.config_file = "ai_monitor_config.json"
        self.load_config()
        
        logging.info("AI Model Monitor initialized")
    
    def load_config(self):
        """Load monitoring configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.monitoring_enabled = config.get('monitoring_enabled', True)
                    self.optimization_threshold = config.get('optimization_threshold', 0.5)
                    self.evaluation_interval_hours = config.get('evaluation_interval_hours', 6)
                    self.min_optimization_interval_hours = config.get('min_optimization_interval_hours', 24)
                logging.info(f"Loaded monitoring config: {config}")
            except Exception as e:
                logging.error(f"Error loading config: {e}")
        else:
            self.save_config()
    
    def save_config(self):
        """Save monitoring configuration"""
        config = {
            'monitoring_enabled': self.monitoring_enabled,
            'optimization_threshold': self.optimization_threshold,
            'evaluation_interval_hours': self.evaluation_interval_hours,
            'min_optimization_interval_hours': self.min_optimization_interval_hours,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def should_optimize_stock(self, symbol, performance_data):
        """Determine if a stock's model should be optimized"""
        if not performance_data:
            return False
        
        # Check performance threshold
        if performance_data['performance_score'] >= self.optimization_threshold:
            return False
        
        # Check if we've optimized this stock recently
        last_optimizations = self.ai_optimizer.optimization_history.get('optimizations', [])
        
        for opt in reversed(last_optimizations):
            if opt['symbol'] == symbol:
                opt_time = datetime.fromisoformat(opt['timestamp'])
                hours_since = (datetime.now() - opt_time).total_seconds() / 3600
                
                if hours_since < self.min_optimization_interval_hours:
                    logging.info(f"Skipping {symbol} - optimized {hours_since:.1f} hours ago")
                    return False
                break
        
        return True
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring and optimization cycle"""
        if not self.monitoring_enabled:
            logging.info("Monitoring disabled - skipping cycle")
            return
        
        logging.info("Starting AI monitoring cycle")
        
        try:
            # Load portfolio stocks
            portfolio_file = "portfolio_universe.json"
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    data = json.load(f)
                    symbols = data.get('stocks', [])
            else:
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']
                logging.warning("Using default symbols - portfolio file not found")
            
            # Evaluate each symbol
            evaluation_results = []
            optimization_results = []
            
            for symbol in symbols[:15]:  # Limit to prevent excessive processing
                try:
                    logging.info(f"Evaluating {symbol}...")
                    
                    # Evaluate performance
                    performance = self.ai_optimizer.evaluate_model_performance(symbol, days_back=14)
                    
                    if performance:
                        evaluation_results.append(performance)
                        
                        # Log performance
                        logging.info(f"{symbol} Performance Score: {performance['performance_score']:.3f}")
                        
                        # Check if optimization is needed
                        if self.should_optimize_stock(symbol, performance):
                            logging.warning(f"{symbol} needs optimization - starting process...")
                            
                            optimization = self.ai_optimizer.optimize_model_parameters(symbol, performance)
                            
                            if optimization:
                                optimization_results.append(optimization)
                                logging.info(f"✅ {symbol} optimized - improvement: {optimization['improvement']:+.3f}")
                            else:
                                logging.error(f"❌ {symbol} optimization failed")
                        else:
                            logging.info(f"✅ {symbol} performing well")
                    
                    # Small delay between evaluations
                    time.sleep(2)
                    
                except Exception as e:
                    logging.error(f"Error evaluating {symbol}: {e}")
                    continue
            
            # Log cycle summary
            total_evaluated = len(evaluation_results)
            total_optimized = len(optimization_results)
            avg_score = sum(r['performance_score'] for r in evaluation_results) / total_evaluated if total_evaluated > 0 else 0
            
            logging.info(f"Monitoring cycle complete:")
            logging.info(f"  - Evaluated: {total_evaluated} stocks")
            logging.info(f"  - Optimized: {total_optimized} models")
            logging.info(f"  - Average Performance: {avg_score:.3f}")
            
            # Save monitoring log
            self.save_monitoring_log(evaluation_results, optimization_results)
            
        except Exception as e:
            logging.error(f"Error in monitoring cycle: {e}")
    
    def save_monitoring_log(self, evaluation_results, optimization_results):
        """Save monitoring cycle results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': evaluation_results,
            'optimization_results': optimization_results,
            'summary': {
                'total_evaluated': len(evaluation_results),
                'total_optimized': len(optimization_results),
                'avg_performance': sum(r['performance_score'] for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0
            }
        }
        
        # Load existing logs
        log_file = "ai_monitoring_log.json"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = {'monitoring_cycles': []}
        
        # Add new log entry
        logs['monitoring_cycles'].append(log_entry)
        
        # Keep only last 100 cycles
        logs['monitoring_cycles'] = logs['monitoring_cycles'][-100:]
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2, default=str)
    
    def start_monitoring(self):
        """Start the monitoring service"""
        logging.info("Starting AI Model Monitoring Service")
        
        # Run initial cycle
        self.run_monitoring_cycle()
        
        # Keep running with intervals
        while self.monitoring_enabled:
            try:
                # Wait for the specified interval
                sleep_time = self.evaluation_interval_hours * 3600  # Convert to seconds
                
                logging.info(f"Next monitoring cycle in {self.evaluation_interval_hours} hours...")
                
                # Sleep in smaller chunks to allow for interruption
                for _ in range(int(sleep_time // 60)):  # Check every minute
                    if not self.monitoring_enabled:
                        break
                    time.sleep(60)
                
                if self.monitoring_enabled:
                    self.run_monitoring_cycle()
                    
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
        
        logging.info("AI Model Monitoring Service stopped")
    
    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.monitoring_enabled = False
        self.save_config()
        logging.info("AI Model Monitoring Service stopping...")

def run_monitoring_service():
    """Run the monitoring service"""
    monitor = AIModelMonitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
        monitor.stop_monitoring()
    except Exception as e:
        logging.error(f"Monitoring service error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    print("🧠 AI Model Monitor Starting...")
    print("Press Ctrl+C to stop")
    run_monitoring_service()
