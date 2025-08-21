#!/usr/bin/env python3
"""
FULLY AUTOMATED SIGNAL OPTIMIZATION RUNNER
Runs continuously to find the best trading signal parameters
No user interaction required - completely automated
"""

import subprocess
import time
import json
import os
from datetime import datetime
import sys

class AutomatedRunner:
    def __init__(self):
        self.optimization_log = "optimization_log.txt"
        self.best_results_file = "best_signal_config.json"
        
    def log_message(self, message):
        """Log messages to both console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        
        # Also write to log file
        with open(self.optimization_log, "a") as f:
            f.write(log_entry + "\n")
    
    def run_optimization_cycle(self, cycle_number):
        """Run one complete optimization cycle"""
        self.log_message(f"🚀 Starting Optimization Cycle #{cycle_number}")
        
        try:
            # Run the optimization script
            result = subprocess.run([
                sys.executable, "automated_signal_optimizer.py"
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                self.log_message(f"✅ Cycle #{cycle_number} completed successfully")
                
                # Check if we have new results
                self.check_for_improvements()
                
                return True
            else:
                self.log_message(f"❌ Cycle #{cycle_number} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_message(f"⏰ Cycle #{cycle_number} timed out after 30 minutes")
            return False
        except Exception as e:
            self.log_message(f"💥 Cycle #{cycle_number} crashed: {str(e)}")
            return False
    
    def check_for_improvements(self):
        """Check if we found better results and save them"""
        # Look for the latest results file
        import glob
        
        result_files = glob.glob("signal_optimization_results_*.json")
        if not result_files:
            return
        
        # Get the most recent file
        latest_file = max(result_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                new_results = json.load(f)
            
            # Check if this is better than our current best
            current_best_performance = -float('inf')
            
            if os.path.exists(self.best_results_file):
                with open(self.best_results_file, 'r') as f:
                    current_best = json.load(f)
                    current_best_performance = current_best.get('best_avg_performance', -float('inf'))
            
            new_performance = new_results.get('best_avg_performance', -float('inf'))
            
            if new_performance > current_best_performance:
                # Save new best results
                with open(self.best_results_file, 'w') as f:
                    json.dump(new_results, f, indent=2)
                
                self.log_message(f"🎯 NEW BEST PERFORMANCE: {new_performance:+.1f}% avg outperformance!")
                self.log_message(f"💾 Saved to {self.best_results_file}")
            else:
                self.log_message(f"📊 Performance: {new_performance:+.1f}% (no improvement)")
                
        except Exception as e:
            self.log_message(f"❌ Error checking results: {str(e)}")
    
    def run_continuous_optimization(self, max_cycles=100, delay_minutes=5):
        """Run optimization continuously"""
        self.log_message("🤖 STARTING FULLY AUTOMATED SIGNAL OPTIMIZATION")
        self.log_message("=" * 60)
        self.log_message(f"🎯 Will run up to {max_cycles} cycles")
        self.log_message(f"⏱️  {delay_minutes} minute delay between cycles")
        self.log_message(f"📁 Results will be saved to {self.best_results_file}")
        self.log_message(f"📝 Logs will be written to {self.optimization_log}")
        self.log_message("=" * 60)
        
        successful_cycles = 0
        failed_cycles = 0
        
        for cycle in range(1, max_cycles + 1):
            self.log_message(f"\n📊 CYCLE {cycle}/{max_cycles}")
            self.log_message("-" * 30)
            
            success = self.run_optimization_cycle(cycle)
            
            if success:
                successful_cycles += 1
            else:
                failed_cycles += 1
            
            # Status update
            self.log_message(f"📈 Success Rate: {successful_cycles}/{cycle} ({successful_cycles/cycle*100:.1f}%)")
            
            # Check if we should continue
            if cycle < max_cycles:
                self.log_message(f"⏳ Waiting {delay_minutes} minutes before next cycle...")
                time.sleep(delay_minutes * 60)
        
        # Final summary
        self.log_message("\n" + "="*60)
        self.log_message("🏁 AUTOMATED OPTIMIZATION COMPLETE")
        self.log_message("="*60)
        self.log_message(f"📊 Total Cycles: {max_cycles}")
        self.log_message(f"✅ Successful: {successful_cycles}")
        self.log_message(f"❌ Failed: {failed_cycles}")
        self.log_message(f"📈 Success Rate: {successful_cycles/max_cycles*100:.1f}%")
        
        if os.path.exists(self.best_results_file):
            with open(self.best_results_file, 'r') as f:
                best_results = json.load(f)
            
            self.log_message(f"🏆 BEST PERFORMANCE: {best_results['best_avg_performance']:+.1f}% avg outperformance")
            self.log_message(f"💾 Best config saved in {self.best_results_file}")
        
        return successful_cycles, failed_cycles

def main():
    """Main entry point for fully automated optimization"""
    runner = AutomatedRunner()
    
    print("🤖 FULLY AUTOMATED TRADING SIGNAL OPTIMIZATION")
    print("=" * 55)
    print("This will run continuously without user input.")
    print("Press Ctrl+C to stop at any time.")
    print("=" * 55)
    
    try:
        # Run with reasonable defaults
        runner.run_continuous_optimization(
            max_cycles=20,      # 20 optimization cycles
            delay_minutes=2     # 2 minutes between cycles
        )
    except KeyboardInterrupt:
        print("\n\n⏹️  OPTIMIZATION STOPPED BY USER")
        print("📊 Check optimization_log.txt for results")
        print("💾 Best config (if found) is in best_signal_config.json")

if __name__ == "__main__":
    main()
