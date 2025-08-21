#!/usr/bin/env python3
"""
OVERNIGHT SIGNAL OPTIMIZATION RUNNER
Special script optimized for long-running overnight optimization
Includes enhanced monitoring, error recovery, and progress tracking
"""

import sys
import os
import time
import signal
from datetime import datetime, timedelta
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from automated_signal_optimizer import AutomatedSignalOptimizer

class OvernightOptimizer:
    def __init__(self):
        self.optimizer = None
        self.start_time = None
        self.graceful_shutdown = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 SHUTDOWN SIGNAL RECEIVED ({signum})")
        print("📊 Saving current progress...")
        self.graceful_shutdown = True
        
        if self.optimizer and self.optimizer.best_config:
            self.save_emergency_results()
        
        print("✅ Graceful shutdown complete!")
        sys.exit(0)
    
    def save_emergency_results(self):
        """Save results in case of interruption"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"emergency_optimization_results_{timestamp}.json"
        
        emergency_data = {
            'interruption_time': datetime.now().isoformat(),
            'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
            'best_config': self.optimizer.best_config,
            'best_performance': self.optimizer.best_avg_performance,
            'completed_iterations': len(self.optimizer.results_history),
            'status': 'INTERRUPTED_BUT_SAVED'
        }
        
        with open(filename, 'w') as f:
            json.dump(emergency_data, f, indent=2, default=str)
        
        print(f"🚨 Emergency results saved to {filename}")
    
    def run_overnight_optimization(self):
        """Run the main overnight optimization"""
        print("🌙 OVERNIGHT SIGNAL OPTIMIZATION STARTING")
        print("=" * 70)
        print(f"🕒 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target: Find optimal parameters across 25 stocks")
        print(f"📊 Using 2-year split-adjusted historical data")
        print(f"🔄 Testing 500+ parameter combinations")
        print(f"💾 Auto-saving progress every 50 iterations")
        print(f"⚡ Press Ctrl+C for graceful shutdown with results saved")
        print("=" * 70)
        
        self.start_time = datetime.now()
        
        try:
            # Initialize optimizer
            print("🚀 Initializing optimizer...")
            self.optimizer = AutomatedSignalOptimizer()
            
            # Run the optimization
            print("🔥 Starting optimization process...")
            best_config = self.optimizer.optimize_signals(max_iterations=500)
            
            if best_config:
                print(f"\n🏆 OVERNIGHT OPTIMIZATION COMPLETE!")
                print("=" * 50)
                
                # Calculate total runtime
                total_runtime = datetime.now() - self.start_time
                hours = total_runtime.total_seconds() / 3600
                
                print(f"⏰ Total runtime: {hours:.1f} hours")
                print(f"🎯 Best average outperformance: {best_config['avg_outperformance']:+.1f}%")
                print(f"📊 Win rate: {best_config['win_rate']:.1%}")
                print(f"🔄 Total configurations tested: {len(self.optimizer.results_history):,}")
                
                # Test on live data
                print(f"\n🧪 Testing best configuration on recent data...")
                live_results = self.optimizer.test_best_config_live()
                
                # Final summary
                self.print_final_summary(best_config, hours)
                
                return best_config
            else:
                print("❌ No successful configurations found!")
                return None
                
        except KeyboardInterrupt:
            print(f"\n🛑 Optimization interrupted by user")
            if self.optimizer and self.optimizer.best_config:
                self.save_emergency_results()
            return None
            
        except Exception as e:
            print(f"❌ Error during optimization: {str(e)}")
            if self.optimizer and self.optimizer.best_config:
                self.save_emergency_results()
            raise
    
    def print_final_summary(self, best_config, runtime_hours):
        """Print comprehensive final summary"""
        print(f"\n" + "="*70)
        print(f"🎉 OVERNIGHT OPTIMIZATION SUCCESS!")
        print(f"="*70)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   🏆 Best Average Outperformance: {best_config['avg_outperformance']:+.1f}%")
        print(f"   🎯 Win Rate: {best_config['win_rate']:.1%}")
        print(f"   📈 Winning Stocks: {best_config['winning_stocks']}/{best_config['total_stocks']}")
        
        print(f"\n⚙️  OPTIMAL PARAMETERS:")
        for key, value in best_config['config'].items():
            print(f"   {key}: {value}")
        
        print(f"\n📊 OPTIMIZATION STATS:")
        print(f"   ⏰ Runtime: {runtime_hours:.1f} hours")
        print(f"   🔄 Configurations tested: {len(self.optimizer.results_history):,}")
        print(f"   📈 Success rate: {len([r for r in self.optimizer.results_history if r['avg_outperformance'] > 0])}/{len(self.optimizer.results_history)}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   ✅ Deploy these optimal parameters to your trading system")
        print(f"   📊 Monitor performance in live trading")
        print(f"   🔄 Re-optimize quarterly with new market data")
        
        print(f"\n💾 RESULTS SAVED:")
        print(f"   📄 Check signal_optimization_results_*.json for full details")
        print(f"   📊 Checkpoint files available for progress tracking")
        
        print(f"\n" + "="*70)

def main():
    """Main entry point for overnight optimization"""
    print("🌙 OVERNIGHT SIGNAL OPTIMIZATION SYSTEM")
    print("🚀 Optimized for 4-8 hour runtime with maximum parameter exploration")
    
    # Create and run overnight optimizer
    overnight_optimizer = OvernightOptimizer()
    
    try:
        best_config = overnight_optimizer.run_overnight_optimization()
        
        if best_config:
            print(f"\n✅ Optimization completed successfully!")
            print(f"🎯 Best configuration saved and ready for deployment")
        else:
            print(f"\n⚠️  Optimization completed with limited results")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        print(f"📊 Check logs and emergency save files for partial results")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
