#!/usr/bin/env python3
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
