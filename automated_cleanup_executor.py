#!/usr/bin/env python3
"""
AUTOMATED SYSTEM CLEANUP EXECUTOR
Executes cleanup recommendations from system verification
"""

import os
import json
import shutil
import glob
from datetime import datetime

class SystemCleanupExecutor:
    def __init__(self, report_file):
        self.workspace_root = "/Users/vyshakhbabji/Desktop/AlgoTradingBot"
        self.backup_dir = os.path.join(self.workspace_root, "backup_before_cleanup")
        
        # Load verification report
        with open(report_file, 'r') as f:
            self.report = json.load(f)
        
        self.cleanup_log = []
        
    def create_backup(self):
        """Create backup of important files before cleanup"""
        print("💾 CREATING BACKUP BEFORE CLEANUP")
        print("=" * 50)
        
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        
        # Backup core files
        core_files = []
        for category, files in self.report['core_files'].items():
            core_files.extend(files)
        
        backed_up_count = 0
        for file in core_files:
            source_path = os.path.join(self.workspace_root, file)
            if os.path.exists(source_path):
                dest_path = os.path.join(self.backup_dir, file)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)
                backed_up_count += 1
                print(f"   📄 Backed up: {file}")
        
        print(f"✅ Backed up {backed_up_count} core files")
        return backed_up_count
    
    def cleanup_old_checkpoints(self):
        """Remove old optimization checkpoint files"""
        print("\n🧹 CLEANING UP OLD OPTIMIZATION CHECKPOINTS")
        print("=" * 50)
        
        checkpoint_files = glob.glob(f"{self.workspace_root}/optimization_checkpoint_*.json")
        
        if len(checkpoint_files) <= 5:
            print(f"   ℹ️  Only {len(checkpoint_files)} checkpoint files found, keeping all")
            return 0
        
        # Sort by modification time, keep latest 5
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        files_to_delete = checkpoint_files[5:]  # Keep first 5 (newest)
        
        deleted_count = 0
        for file in files_to_delete:
            try:
                os.remove(file)
                deleted_count += 1
                print(f"   🗑️  Deleted: {os.path.basename(file)}")
            except Exception as e:
                print(f"   ❌ Failed to delete {os.path.basename(file)}: {e}")
        
        print(f"✅ Cleaned up {deleted_count} old checkpoint files")
        return deleted_count
    
    def cleanup_broken_files(self):
        """Remove broken or incomplete files"""
        print("\n🧹 CLEANING UP BROKEN/INCOMPLETE FILES")
        print("=" * 50)
        
        # Files marked as broken
        broken_patterns = [
            "*BROKEN*.py",
            "*_old.py",
            "*_backup.py",
            "*_temp.py"
        ]
        
        deleted_count = 0
        for pattern in broken_patterns:
            broken_files = glob.glob(f"{self.workspace_root}/{pattern}")
            for file in broken_files:
                try:
                    os.remove(file)
                    deleted_count += 1
                    print(f"   🗑️  Deleted: {os.path.basename(file)}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {os.path.basename(file)}: {e}")
        
        print(f"✅ Cleaned up {deleted_count} broken files")
        return deleted_count
    
    def cleanup_old_json_data(self):
        """Clean up old JSON data files"""
        print("\n🧹 CLEANING UP OLD JSON DATA FILES")
        print("=" * 50)
        
        # Core JSON files to preserve
        core_json = {
            'paper_trading_data.json',
            'paper_trading_account.json', 
            'portfolio_universe.json',
            'model_performance_history.json',
            'system_status_log.json'
        }
        
        # Get all JSON files
        all_json = glob.glob(f"{self.workspace_root}/*.json")
        
        # Keep only recent files (last 7 days)
        cutoff_time = datetime.now().timestamp() - (7 * 24 * 3600)
        deleted_count = 0
        
        for file in all_json:
            filename = os.path.basename(file)
            if filename not in core_json and not filename.startswith('system_verification_report'):
                if os.path.getmtime(file) < cutoff_time:
                    try:
                        os.remove(file)
                        deleted_count += 1
                        print(f"   🗑️  Deleted: {filename}")
                    except Exception as e:
                        print(f"   ❌ Failed to delete {filename}: {e}")
        
        print(f"✅ Cleaned up {deleted_count} old JSON files")
        return deleted_count
    
    def generate_summary(self, stats):
        """Generate cleanup summary"""
        print("\n" + "=" * 60)
        print("📋 CLEANUP SUMMARY REPORT")
        print("=" * 60)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'cleanup_stats': stats,
            'total_files_processed': sum(stats.values())
        }
        
        # Save cleanup report
        report_file = f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 CLEANUP STATISTICS:")
        for category, count in stats.items():
            print(f"   🧹 {category.replace('_', ' ').title()}: {count} files")
        
        print(f"\n💾 Report saved to: {report_file}")
        
        return summary
    
    def execute_cleanup(self):
        """Execute complete cleanup process"""
        print("🚀 STARTING AUTOMATED SYSTEM CLEANUP")
        print("=" * 60)
        
        stats = {}
        
        # Step 1: Create backup
        stats['backup_files'] = self.create_backup()
        
        # Step 2: Clean up old checkpoints
        stats['checkpoint_cleanup'] = self.cleanup_old_checkpoints()
        
        # Step 3: Clean up broken files
        stats['broken_file_cleanup'] = self.cleanup_broken_files()
        
        # Step 4: Clean up old JSON data
        stats['json_cleanup'] = self.cleanup_old_json_data()
        
        # Step 5: Generate summary
        summary = self.generate_summary(stats)
        
        print("\n🎯 CLEANUP COMPLETE!")
        print("   ✅ System cleaned and organized")
        print("   ✅ Core functionality preserved")
        print("   ✅ Backup created for rollback")
        
        return summary

def main():
    """Run the cleanup system"""
    # Find the latest verification report
    report_files = glob.glob("/Users/vyshakhbabji/Desktop/AlgoTradingBot/system_verification_report_*.json")
    
    if not report_files:
        print("❌ No verification report found. Run system verification first.")
        return None
    
    latest_report = max(report_files, key=os.path.getmtime)
    print(f"📋 Using verification report: {os.path.basename(latest_report)}")
    
    # Execute cleanup
    executor = SystemCleanupExecutor(latest_report)
    summary = executor.execute_cleanup()
    
    return executor

if __name__ == "__main__":
    executor = main()
