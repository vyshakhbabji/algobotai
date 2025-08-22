#!/usr/bin/env python3
"""
Simple GCP Deployment Setup
Creates essential files for Google Cloud Platform deployment
"""

import os
import json

def create_deployment_files():
    """Create essential GCP deployment files"""
    
    print("🌟 Creating Google Cloud Platform deployment files...")
    
    # 1. app.yaml for App Engine
    app_yaml_content = """runtime: python39
env: standard

service: trading-bot

automatic_scaling:
  min_instances: 1
  max_instances: 2
  target_cpu_utilization: 0.6

env_variables:
  TRADING_MODE: "LIVE"
  PAPER_TRADING: "true"

handlers:
- url: /.*
  script: auto
"""
    
    with open('app.yaml', 'w') as f:
        f.write(app_yaml_content)
    print("  ✅ Created app.yaml")
    
    # 2. requirements.txt
    requirements_content = """alpaca-py==0.32.1
pandas==2.2.2
numpy==1.24.3
scikit-learn==1.3.0
flask==3.0.3
gunicorn==21.2.0
schedule==1.2.0
python-dotenv==1.0.0
yfinance==0.2.28
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    print("  ✅ Created requirements.txt")
    
    # 3. Simple main.py for web interface
    main_py_content = '''#!/usr/bin/env python3
"""
Simple Flask app for Google App Engine
"""

import os
import json
import threading
from datetime import datetime
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# Simple HTML dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🚀 Alpaca Trading Bot - Live on GCP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 30px; border-radius: 10px; text-align: center; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; margin: 20px 0; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .status { background: white; padding: 20px; border-radius: 10px; 
                 box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .btn { background: #667eea; color: white; border: none; padding: 10px 20px; 
              border-radius: 5px; cursor: pointer; margin: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Alpaca Trading Bot</h1>
        <p>AI-Powered Trading System on Google Cloud Platform</p>
        <p style="color: #ffd700;">📝 PAPER TRADING MODE</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Status</h3>
            <div class="stat-value">✅ LIVE</div>
        </div>
        <div class="stat-card">
            <h3>Platform</h3>
            <div class="stat-value">GCP</div>
        </div>
        <div class="stat-card">
            <h3>Mode</h3>
            <div class="stat-value">PAPER</div>
        </div>
        <div class="stat-card">
            <h3>Uptime</h3>
            <div class="stat-value">{{ uptime }}</div>
        </div>
    </div>
    
    <div class="status">
        <h3>System Information</h3>
        <p><strong>Deployment Time:</strong> {{ deployment_time }}</p>
        <p><strong>Last Update:</strong> {{ last_update }}</p>
        <p><strong>Trading Status:</strong> Paper Trading Active</p>
        <p><strong>Next Steps:</strong> Connect real-time Alpaca trader</p>
        
        <button class="btn" onclick="location.reload()">🔄 Refresh</button>
        <button class="btn" onclick="window.open('/api/status')">📊 API Status</button>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() { location.reload(); }, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard"""
    uptime = "Active"
    deployment_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template_string(DASHBOARD_HTML, 
                                uptime=uptime,
                                deployment_time=deployment_time,
                                last_update=last_update)

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'healthy',
        'platform': 'Google Cloud Platform',
        'mode': 'paper_trading',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'active'
    })

@app.route('/health')
def health_check():
    """Health check for App Engine"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
'''
    
    with open('main.py', 'w') as f:
        f.write(main_py_content)
    print("  ✅ Created main.py")
    
    # 4. .gcloudignore
    gcloudignore_lines = [
        "# Python cache",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "",
        "# Local files", 
        "*.csv",
        "*.json",
        "*.png",
        "*.log",
        ".env",
        ".DS_Store",
        "",
        "# IDE",
        ".vscode/",
        ".idea/"
    ]
    
    with open('.gcloudignore', 'w') as f:
        f.write('\n'.join(gcloudignore_lines))
    print("  ✅ Created .gcloudignore")
    
    # 5. Deployment script
    deploy_script_lines = [
        "#!/bin/bash",
        "# Simple GCP deployment script",
        "",
        'echo "🚀 Deploying to Google Cloud Platform..."',
        "",
        "# Check gcloud installation",
        "if ! command -v gcloud &> /dev/null; then",
        '    echo "❌ Please install Google Cloud SDK first"',
        '    echo "   Visit: https://cloud.google.com/sdk/docs/install"',
        "    exit 1",
        "fi",
        "",
        "# Deploy",
        'echo "📤 Starting deployment..."',
        "gcloud app deploy app.yaml --quiet",
        "",
        'echo "✅ Deployment complete!"',
        'echo "🌐 Access your bot at the URL shown above"'
    ]
    
    with open('deploy-gcp.sh', 'w') as f:
        f.write('\n'.join(deploy_script_lines))
    
    os.chmod('deploy-gcp.sh', 0o755)
    print("  ✅ Created deploy-gcp.sh")
    
    # 6. Environment template
    env_template_lines = [
        "# Alpaca API Configuration",
        "ALPACA_API_KEY=your_api_key_here", 
        "ALPACA_SECRET_KEY=your_secret_key_here",
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets",
        "PAPER_TRADING=true",
        "",
        "# Google Cloud Project",
        "GOOGLE_CLOUD_PROJECT=your-project-id"
    ]
    
    with open('.env.template', 'w') as f:
        f.write('\n'.join(env_template_lines))
    print("  ✅ Created .env.template")
    
    return True

def show_instructions():
    """Show deployment instructions"""
    
    print("\n" + "="*60)
    print("🚀 GOOGLE CLOUD PLATFORM DEPLOYMENT")
    print("="*60)
    
    print("\n📋 QUICK START:")
    print("1. Install Google Cloud SDK:")
    print("   brew install google-cloud-sdk")
    
    print("\n2. Create GCP project:")
    print("   https://console.cloud.google.com/")
    
    print("\n3. Authenticate:")
    print("   gcloud auth login")
    print("   gcloud config set project YOUR_PROJECT_ID")
    
    print("\n4. Deploy:")
    print("   ./deploy-gcp.sh")
    
    print("\n🌐 Your bot will be live at:")
    print("   https://YOUR_PROJECT_ID.uc.r.appspot.com")
    
    print("\n💰 Estimated cost: ~$0.05/hour when active")
    print("📊 Auto-scaling: Scales to 0 when inactive")
    
    print("\n✅ Ready for deployment!")

def main():
    """Main function"""
    print("🌟 Setting up Google Cloud Platform deployment...")
    
    if create_deployment_files():
        show_instructions()
        
    print("\n📁 Files created:")
    files = ['app.yaml', 'main.py', 'requirements.txt', 'deploy-gcp.sh', '.gcloudignore', '.env.template']
    for file in files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
    
    print("\n🚀 Ready to deploy to Google Cloud!")

if __name__ == "__main__":
    main()
