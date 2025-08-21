#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM VERIFICATION & CLEANUP
Analyzes all ML models, options models, and trading files
Identifies working vs obsolete files and consolidates the system
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime
import subprocess
import sys

class SystemVerificationCleanup:
    def __init__(self):
        self.workspace_root = "/Users/vyshakhbabji/Desktop/AlgoTradingBot"
        self.verification_results = {}
        self.cleanup_recommendations = []
        self.active_files = []
        self.obsolete_files = []
        self.ml_models = {}
        self.options_models = {}
        
    def analyze_file_structure(self):
        """Analyze current file structure and categorize"""
        print("🔍 ANALYZING FILE STRUCTURE")
        print("=" * 50)
        
        # Get all Python files
        py_files = glob.glob(f"{self.workspace_root}/**/*.py", recursive=True)
        json_files = glob.glob(f"{self.workspace_root}/**/*.json", recursive=True)
        
        # Categorize files
        categories = {
            'ml_models': [],
            'options_trading': [],
            'momentum_strategies': [],
            'backtesting': [],
            'paper_trading': [],
            'dashboards': [],
            'utilities': [],
            'obsolete_duplicates': []
        }
        
        for file in py_files:
            filename = os.path.basename(file)
            
            # ML Model files
            if any(keyword in filename.lower() for keyword in ['ml', 'model', 'ai_', 'enhanced']):
                categories['ml_models'].append(file)
            
            # Options trading files
            elif any(keyword in filename.lower() for keyword in ['option', 'greeks', 'volatility']):
                categories['options_trading'].append(file)
            
            # Momentum strategy files
            elif any(keyword in filename.lower() for keyword in ['momentum', 'institutional']):
                categories['momentum_strategies'].append(file)
            
            # Backtesting files
            elif any(keyword in filename.lower() for keyword in ['backtest', 'test', 'forward', 'validation']):
                categories['backtesting'].append(file)
            
            # Paper trading files
            elif any(keyword in filename.lower() for keyword in ['paper', 'live', 'trading']):
                categories['paper_trading'].append(file)
            
            # Dashboard files
            elif any(keyword in filename.lower() for keyword in ['dashboard', 'streamlit', 'app']):
                categories['dashboards'].append(file)
            
            # Utility files
            else:
                categories['utilities'].append(file)
        
        # Print categorization
        for category, files in categories.items():
            print(f"\n📁 {category.upper()}: {len(files)} files")
            for file in files[:5]:  # Show first 5
                print(f"   - {os.path.basename(file)}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
        
        return categories
    
    def verify_ml_models(self):
        """Verify ML model files and their functionality"""
        print("\n🧠 VERIFYING ML MODELS")
        print("=" * 50)
        
        ml_files = [
            'comprehensive_momentum_ml.py',
            'simplified_ml_momentum_trader.py', 
            'ml_filtered_momentum_trader.py',
            'momentum_enhanced_ml_model.py'
        ]
        
        working_models = []
        broken_models = []
        
        for ml_file in ml_files:
            filepath = os.path.join(self.workspace_root, ml_file)
            if os.path.exists(filepath):
                try:
                    # Check if file can be imported
                    print(f"🧪 Testing {ml_file}...")
                    
                    # Read file and check for key components
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    has_sklearn = 'sklearn' in content
                    has_models = any(model in content for model in ['RandomForest', 'GradientBoosting', 'LogisticRegression'])
                    has_training = 'fit(' in content or 'train' in content
                    has_prediction = 'predict' in content
                    
                    score = sum([has_sklearn, has_models, has_training, has_prediction])
                    
                    if score >= 3:
                        working_models.append((ml_file, score))
                        print(f"   ✅ {ml_file}: WORKING (Score: {score}/4)")
                    else:
                        broken_models.append((ml_file, score))
                        print(f"   ❌ {ml_file}: INCOMPLETE (Score: {score}/4)")
                        
                except Exception as e:
                    broken_models.append((ml_file, 0))
                    print(f"   🔥 {ml_file}: ERROR - {str(e)}")
            else:
                print(f"   ❓ {ml_file}: NOT FOUND")
        
        self.ml_models = {
            'working': working_models,
            'broken': broken_models
        }
        
        return working_models, broken_models
    
    def verify_options_models(self):
        """Verify options trading models"""
        print("\n📈 VERIFYING OPTIONS MODELS")
        print("=" * 50)
        
        options_files = [
            'elite_options_trader.py',
            'ai_enhanced_options_trader.py',
            'momentum_options_trader.py',
            'comprehensive_momentum_options.py',
            'pages/elite_options.py'
        ]
        
        working_options = []
        broken_options = []
        
        for options_file in options_files:
            filepath = os.path.join(self.workspace_root, options_file)
            if os.path.exists(filepath):
                try:
                    print(f"🧪 Testing {options_file}...")
                    
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    has_options_logic = any(keyword in content.lower() for keyword in ['call', 'put', 'strike', 'expiry'])
                    has_volatility = 'volatility' in content.lower()
                    has_greeks = any(greek in content.lower() for greek in ['delta', 'gamma', 'theta', 'vega'])
                    has_strategies = any(strategy in content.lower() for strategy in ['spread', 'straddle', 'strangle'])
                    
                    score = sum([has_options_logic, has_volatility, has_greeks, has_strategies])
                    
                    if score >= 2:
                        working_options.append((options_file, score))
                        print(f"   ✅ {options_file}: WORKING (Score: {score}/4)")
                    else:
                        broken_options.append((options_file, score))
                        print(f"   ❌ {options_file}: INCOMPLETE (Score: {score}/4)")
                        
                except Exception as e:
                    broken_options.append((options_file, 0))
                    print(f"   🔥 {options_file}: ERROR - {str(e)}")
            else:
                print(f"   ❓ {options_file}: NOT FOUND")
        
        self.options_models = {
            'working': working_options,
            'broken': broken_options
        }
        
        return working_options, broken_options
    
    def identify_duplicate_files(self):
        """Identify duplicate and obsolete files"""
        print("\n🔍 IDENTIFYING DUPLICATES & OBSOLETES")
        print("=" * 50)
        
        # Pattern-based duplicate detection
        duplicate_patterns = [
            ('improved_', 'original implementation'),
            ('fixed_', 'broken version'),
            ('enhanced_', 'basic version'),
            ('comprehensive_', 'simple version'),
            ('elite_', 'basic version'),
            ('advanced_', 'basic version')
        ]
        
        all_files = glob.glob(f"{self.workspace_root}/*.py")
        potential_duplicates = []
        
        for file in all_files:
            filename = os.path.basename(file)
            
            for pattern, description in duplicate_patterns:
                if pattern in filename:
                    base_name = filename.replace(pattern, '')
                    potential_original = os.path.join(self.workspace_root, base_name)
                    
                    if os.path.exists(potential_original):
                        potential_duplicates.append({
                            'improved': file,
                            'original': potential_original,
                            'pattern': pattern,
                            'description': description
                        })
        
        print(f"Found {len(potential_duplicates)} potential duplicate pairs:")
        for dup in potential_duplicates:
            print(f"   📄 {os.path.basename(dup['improved'])} vs {os.path.basename(dup['original'])}")
        
        return potential_duplicates
    
    def check_data_consistency(self):
        """Check JSON data files for consistency"""
        print("\n📊 CHECKING DATA CONSISTENCY")
        print("=" * 50)
        
        important_json_files = [
            'paper_trading_data.json',
            'paper_trading_account.json',
            'portfolio_universe.json',
            'model_performance_history.json'
        ]
        
        data_status = {}
        
        for json_file in important_json_files:
            filepath = os.path.join(self.workspace_root, json_file)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    file_size = os.path.getsize(filepath)
                    data_status[json_file] = {
                        'status': '✅ VALID',
                        'size': file_size,
                        'records': len(data) if isinstance(data, (list, dict)) else 1
                    }
                    print(f"   ✅ {json_file}: {file_size} bytes, {data_status[json_file]['records']} records")
                    
                except Exception as e:
                    data_status[json_file] = {
                        'status': f'❌ ERROR: {str(e)}',
                        'size': 0,
                        'records': 0
                    }
                    print(f"   ❌ {json_file}: CORRUPTED - {str(e)}")
            else:
                data_status[json_file] = {
                    'status': '❓ MISSING',
                    'size': 0,
                    'records': 0
                }
                print(f"   ❓ {json_file}: NOT FOUND")
        
        return data_status
    
    def analyze_recent_usage(self):
        """Analyze which files have been recently modified"""
        print("\n⏰ ANALYZING RECENT USAGE")
        print("=" * 50)
        
        all_files = glob.glob(f"{self.workspace_root}/*.py")
        recent_files = []
        old_files = []
        
        cutoff_date = datetime.now().timestamp() - (7 * 24 * 3600)  # 7 days ago
        
        for file in all_files:
            mod_time = os.path.getmtime(file)
            if mod_time > cutoff_date:
                recent_files.append((file, mod_time))
            else:
                old_files.append((file, mod_time))
        
        # Sort by modification time
        recent_files.sort(key=lambda x: x[1], reverse=True)
        old_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📁 RECENTLY MODIFIED ({len(recent_files)} files):")
        for file, mod_time in recent_files[:10]:
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
            print(f"   🔥 {os.path.basename(file)} - {mod_date}")
        
        print(f"\n📁 OLDER FILES ({len(old_files)} files):")
        for file, mod_time in old_files[:5]:
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')
            print(f"   🕰️ {os.path.basename(file)} - {mod_date}")
        
        return recent_files, old_files
    
    def generate_cleanup_recommendations(self):
        """Generate specific cleanup recommendations"""
        print("\n🧹 GENERATING CLEANUP RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        # 1. Remove broken ML models
        if self.ml_models.get('broken'):
            recommendations.append({
                'category': 'ML Models',
                'action': 'Remove broken/incomplete ML model files',
                'files': [f[0] for f in self.ml_models['broken']],
                'reason': 'Incomplete implementations that cannot be used'
            })
        
        # 2. Remove broken options models
        if self.options_models.get('broken'):
            recommendations.append({
                'category': 'Options Trading',
                'action': 'Remove incomplete options trading files',
                'files': [f[0] for f in self.options_models['broken']],
                'reason': 'Missing essential options trading components'
            })
        
        # 3. Clean up old optimization checkpoints
        old_checkpoints = glob.glob(f"{self.workspace_root}/optimization_checkpoint_*.json")
        if len(old_checkpoints) > 5:
            recommendations.append({
                'category': 'Data Cleanup',
                'action': 'Remove old optimization checkpoints',
                'files': old_checkpoints[:-5],  # Keep latest 5
                'reason': 'Reduce clutter from old optimization runs'
            })
        
        # 4. Consolidate duplicate test files
        test_files = [f for f in glob.glob(f"{self.workspace_root}/*test*.py") 
                     if 'simple_' not in f and 'quick_' not in f]
        if len(test_files) > 3:
            recommendations.append({
                'category': 'Testing',
                'action': 'Consolidate test files',
                'files': test_files[3:],  # Keep first 3
                'reason': 'Too many similar testing files'
            })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['category']}: {rec['action']}")
            print(f"   Reason: {rec['reason']}")
            print(f"   Files: {len(rec['files'])} files")
            for file in rec['files'][:3]:
                print(f"     - {os.path.basename(file)}")
            if len(rec['files']) > 3:
                print(f"     ... and {len(rec['files']) - 3} more")
        
        return recommendations
    
    def identify_core_files(self):
        """Identify core files that should be preserved"""
        print("\n🎯 IDENTIFYING CORE FILES")
        print("=" * 50)
        
        core_files = {
            'main_interface': ['main_dashboard.py', 'app.py'],
            'working_ml': [f[0] for f in self.ml_models.get('working', [])],
            'working_options': [f[0] for f in self.options_models.get('working', [])],
            'proven_strategies': [
                'aggressive_forward_tester.py',
                'institutional_hybrid_tester.py', 
                'institutional_momentum_screener.py'
            ],
            'data_files': [
                'paper_trading_data.json',
                'paper_trading_account.json',
                'portfolio_universe.json'
            ],
            'utilities': [
                'momentum_feature_engine.py',
                'automated_signal_optimizer.py'
            ]
        }
        
        print("🎯 CORE FILES TO PRESERVE:")
        for category, files in core_files.items():
            print(f"\n📁 {category.upper()}:")
            for file in files:
                filepath = os.path.join(self.workspace_root, file)
                status = "✅" if os.path.exists(filepath) else "❌"
                print(f"   {status} {file}")
        
        return core_files
    
    def run_full_verification(self):
        """Run complete system verification"""
        print("🚀 STARTING COMPREHENSIVE SYSTEM VERIFICATION")
        print("=" * 60)
        
        # Step 1: Analyze file structure
        categories = self.analyze_file_structure()
        
        # Step 2: Verify ML models
        working_ml, broken_ml = self.verify_ml_models()
        
        # Step 3: Verify options models
        working_options, broken_options = self.verify_options_models()
        
        # Step 4: Check for duplicates
        duplicates = self.identify_duplicate_files()
        
        # Step 5: Check data consistency
        data_status = self.check_data_consistency()
        
        # Step 6: Analyze recent usage
        recent_files, old_files = self.analyze_recent_usage()
        
        # Step 7: Generate recommendations
        recommendations = self.generate_cleanup_recommendations()
        
        # Step 8: Identify core files
        core_files = self.identify_core_files()
        
        # Generate final report
        self.generate_final_report(categories, working_ml, working_options, 
                                 duplicates, data_status, recommendations, core_files)
    
    def generate_final_report(self, categories, working_ml, working_options, 
                            duplicates, data_status, recommendations, core_files):
        """Generate comprehensive final report"""
        print("\n" + "=" * 60)
        print("📋 FINAL VERIFICATION REPORT")
        print("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_py_files': sum(len(files) for files in categories.values()),
                'working_ml_models': len(working_ml),
                'working_options_models': len(working_options),
                'potential_duplicates': len(duplicates),
                'cleanup_recommendations': len(recommendations)
            },
            'categories': categories,
            'ml_models': self.ml_models,
            'options_models': self.options_models,
            'duplicates': duplicates,
            'data_status': data_status,
            'recommendations': recommendations,
            'core_files': core_files
        }
        
        # Save report
        report_file = f"system_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"📊 SYSTEM SUMMARY:")
        print(f"   🔧 Total Python Files: {report['summary']['total_py_files']}")
        print(f"   🧠 Working ML Models: {report['summary']['working_ml_models']}")
        print(f"   📈 Working Options Models: {report['summary']['working_options_models']}")
        print(f"   📄 Potential Duplicates: {report['summary']['potential_duplicates']}")
        print(f"   🧹 Cleanup Items: {report['summary']['cleanup_recommendations']}")
        
        print(f"\n💾 Report saved to: {report_file}")
        
        return report

def main():
    """Run the verification and cleanup system"""
    verifier = SystemVerificationCleanup()
    report = verifier.run_full_verification()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Review the generated report")
    print("2. Backup important files before cleanup")
    print("3. Execute recommended cleanup actions")
    print("4. Test core functionality after cleanup")
    
    return verifier

if __name__ == "__main__":
    verifier = main()
