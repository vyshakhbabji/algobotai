#!/bin/bash

# AWS Deployment Script for Real-Time Alpaca Trading Bot
# Deploys to AWS EC2 with Docker

set -e

echo "🚀 AWS DEPLOYMENT SCRIPT"
echo "Real-Time Alpaca Trading Bot"
echo "=========================="

# Configuration
AWS_REGION="us-east-1"
INSTANCE_TYPE="t3.small"
KEY_NAME="trading-bot-key"
SECURITY_GROUP="trading-bot-sg"
AMI_ID="ami-0c94855ba95b798c7"  # Amazon Linux 2023

# Function to create security group
create_security_group() {
    echo "🔒 Creating security group..."
    
    # Create security group
    aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "Security group for trading bot" \
        --region $AWS_REGION
    
    # Allow SSH access
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    # Allow monitoring port
    aws ec2 authorize-security-group-ingress \
        --group-name $SECURITY_GROUP \
        --protocol tcp \
        --port 8080 \
        --cidr 0.0.0.0/0 \
        --region $AWS_REGION
    
    echo "✅ Security group created"
}

# Function to create key pair
create_key_pair() {
    echo "🔑 Creating key pair..."
    
    aws ec2 create-key-pair \
        --key-name $KEY_NAME \
        --query 'KeyMaterial' \
        --output text \
        --region $AWS_REGION > ${KEY_NAME}.pem
    
    chmod 400 ${KEY_NAME}.pem
    echo "✅ Key pair created: ${KEY_NAME}.pem"
}

# Function to launch EC2 instance
launch_instance() {
    echo "🖥️ Launching EC2 instance..."
    
    # User data script for instance setup
    cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y docker git
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone trading bot repository (you'll need to setup git access)
# For now, we'll create the necessary files
mkdir -p /home/ec2-user/trading-bot
chown ec2-user:ec2-user /home/ec2-user/trading-bot
EOF

    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --count 1 \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-groups $SECURITY_GROUP \
        --user-data file://user-data.sh \
        --region $AWS_REGION \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    echo "✅ Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    echo "⏳ Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $AWS_REGION
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region $AWS_REGION)
    
    echo "✅ Instance running at: $PUBLIC_IP"
    return $PUBLIC_IP
}

# Function to deploy application
deploy_application() {
    local public_ip=$1
    
    echo "📦 Deploying application to $public_ip..."
    
    # Wait for SSH to be ready
    echo "⏳ Waiting for SSH to be ready..."
    sleep 60
    
    # Copy files to instance
    echo "📁 Copying files to instance..."
    scp -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no \
        realtime_alpaca_trader.py \
        alpaca_config.json \
        alpaca_integration.py \
        requirements.txt \
        Dockerfile \
        docker-compose.yml \
        ec2-user@${public_ip}:/home/ec2-user/trading-bot/
    
    # Connect and start the application
    echo "🚀 Starting application..."
    ssh -i ${KEY_NAME}.pem -o StrictHostKeyChecking=no ec2-user@${public_ip} << 'EOF'
cd /home/ec2-user/trading-bot
sudo docker-compose up -d --build
echo "✅ Trading bot started successfully!"
echo "📊 Monitor at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8080"
EOF
}

# Main deployment process
main() {
    echo "🔍 Checking AWS CLI..."
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Please install it first."
        exit 1
    fi
    
    echo "📋 Deployment Configuration:"
    echo "   Region: $AWS_REGION"
    echo "   Instance Type: $INSTANCE_TYPE"
    echo "   Key Name: $KEY_NAME"
    echo "   Security Group: $SECURITY_GROUP"
    
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Deployment cancelled"
        exit 1
    fi
    
    # Check if files exist
    if [[ ! -f "realtime_alpaca_trader.py" ]] || [[ ! -f "alpaca_config.json" ]]; then
        echo "❌ Required files not found. Run from the trading bot directory."
        exit 1
    fi
    
    # Create AWS resources
    echo "🏗️ Creating AWS resources..."
    create_security_group 2>/dev/null || echo "⚠️ Security group may already exist"
    create_key_pair 2>/dev/null || echo "⚠️ Key pair may already exist"
    
    # Launch and deploy
    public_ip=$(launch_instance)
    deploy_application $public_ip
    
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "========================"
    echo "✅ Trading bot deployed to AWS EC2"
    echo "🌐 Public IP: $public_ip"
    echo "🔗 SSH Access: ssh -i ${KEY_NAME}.pem ec2-user@${public_ip}"
    echo "📊 Monitor: http://${public_ip}:8080"
    echo "📈 Grafana: http://${public_ip}:3000 (admin/admin123)"
    echo ""
    echo "🔧 Next Steps:"
    echo "1. Update alpaca_config.json with your API keys"
    echo "2. Monitor the bot performance"
    echo "3. Check logs: ssh -i ${KEY_NAME}.pem ec2-user@${public_ip} 'docker logs alpaca-trading-bot'"
}

# Error handling
trap 'echo "❌ Deployment failed. Check the error above."' ERR

# Run main function
main "$@"
