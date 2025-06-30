#!/bin/bash

# Modal Auto Caption Generator Deployment Script
echo "🚀 Modal Auto Caption Generator - Deployment Script"
echo "=================================================="

# Check if Modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
else
    echo "✅ Modal CLI found"
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

# Check if OpenAI API key secret exists
echo "🔑 Checking OpenAI API key secret..."
if ! modal secret list | grep -q "openai-api-key"; then
    echo "❌ OpenAI API key secret not found."
    echo "Please create it with:"
    echo "   modal secret create openai-api-key OPENAI_API_KEY=your_actual_api_key_here"
    echo "Then run this script again."
    exit 1
fi

echo "✅ OpenAI API key secret found"

# Deploy the application
echo "📦 Deploying to Modal..."
python modal_deploy.py

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "Your endpoints are now available at:"
echo "- Health Check: https://auto-caption-generator--health-check.modal.run"
echo "- Generate Subtitles: https://auto-caption-generator--generate-subtitles.modal.run"
echo "- Generate Live Subtitles: https://auto-caption-generator--generate-live-subtitles.modal.run"
echo ""
echo "To test your deployment, run:"
echo "   python test_modal_deployment.py https://auto-caption-generator--health-check.modal.run" 