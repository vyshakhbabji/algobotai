#!/bin/bash
# Simple GCP deployment script

echo "🚀 Deploying to Google Cloud Platform..."

# Check gcloud installation
if ! command -v gcloud &> /dev/null; then
    echo "❌ Please install Google Cloud SDK first"
    echo "   Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Deploy
echo "📤 Starting deployment..."
gcloud app deploy app.yaml --quiet

echo "✅ Deployment complete!"
echo "🌐 Access your bot at the URL shown above"