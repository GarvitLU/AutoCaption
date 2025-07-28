#!/bin/bash

echo "🚀 Deploying Auto Caption Generator to Modal..."

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Please install it first:"
    echo "pip install modal"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "modal_app.py" ]; then
    echo "❌ modal_app.py not found. Please run this script from the project directory."
    exit 1
fi

echo "📦 Deploying with new styling features..."
echo "✅ 12 fonts supported"
echo "✅ Advanced styling options (bg_color, font_color, highlight_color)"
echo "✅ Custom font sizes and weights"
echo "✅ Color parsing (hex and named colors)"

# Deploy to Modal
modal deploy modal_app.py

echo "🎉 Deployment complete!"
echo ""
echo "📋 Available endpoints:"
echo "  - POST /generate-subtitles/ (classic subtitle generation)"
echo "  - POST /generate-live-subtitles/ (karaoke-style with advanced styling)"
echo "  - GET /health_check (health check)"
echo ""
echo "🎨 New styling features:"
echo "  - bg_color: Background color (hex or named)"
echo "  - font_color: Text color (hex or named)"
echo "  - highlight_color: Highlight color (hex or named)"
echo "  - font_family: Font selection (12 fonts available)"
echo "  - font_weight: Font weight (bold/regular)"
echo "  - font_size: Font size in pixels"
echo ""
echo "📚 Available fonts: arial, georgia, montserrat, verdana, comic_sans, times_new_roman, courier_new, trebuchet_ms, tahoma, roboto, open_sans, raleway" 