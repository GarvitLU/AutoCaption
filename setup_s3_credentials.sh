#!/bin/bash

# S3 Credentials Setup Script for Modal
echo "🔑 S3 Credentials Setup for Modal"
echo "=================================="

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Please install it first:"
    echo "   pip install modal"
    exit 1
fi

# Check if user is authenticated
echo "🔐 Checking Modal authentication..."
if ! modal token list &> /dev/null; then
    echo "❌ Not authenticated with Modal. Please run:"
    echo "   modal token new"
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Modal authentication confirmed"

# Check if S3 credentials secret already exists
if modal secret list | grep -q "s3-credentials"; then
    echo "⚠️  S3 credentials secret already exists."
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing secret..."
        modal secret delete s3-credentials
    else
        echo "✅ Keeping existing secret"
        exit 0
    fi
fi

# Get S3 credentials from user
echo ""
echo "📝 Please provide your S3 credentials:"
echo ""

read -p "S3 Bucket Name: " S3_BUCKET_NAME
read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -s -p "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo
read -p "AWS Region (default: us-east-1): " AWS_REGION
AWS_REGION=${AWS_REGION:-us-east-1}

# Validate inputs
if [ -z "$S3_BUCKET_NAME" ] || [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "❌ All fields are required!"
    exit 1
fi

# Create the secret
echo ""
echo "🔐 Creating S3 credentials secret..."
modal secret create s3-credentials \
    S3_BUCKET_NAME="$S3_BUCKET_NAME" \
    AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    AWS_REGION="$AWS_REGION"

if [ $? -eq 0 ]; then
    echo "✅ S3 credentials secret created successfully!"
    echo ""
    echo "🎉 You can now deploy your app with S3 upload functionality!"
    echo "Run: ./deploy_to_modal.sh"
else
    echo "❌ Failed to create S3 credentials secret"
    exit 1
fi 