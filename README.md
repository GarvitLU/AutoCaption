# Auto Caption Generator

An advanced video captioning API that automatically generates and embeds captions into videos using OpenAI's Whisper model for transcription. Features word-level highlighting, multiple caption styles, and karaoke-style subtitles.

## Features

- **Automatic Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Word-Level Highlighting**: Highlights individual words as they are spoken (karaoke-style)
- **Multiple Caption Styles**: 9 different visual styles for captions
- **Customizable Parameters**: Font size, color, position, and background options
- **Multi-language Support**: Supports transcription in multiple languages
- **FastAPI-based REST API**: Modern, fast API with automatic documentation
- **Video Processing**: Direct video file processing with embedded captions
- **Temporary File Management**: Automatic cleanup of temporary files

## Setup

### Prerequisites

- Python 3.8+
- ImageMagick (for text rendering)
- FFmpeg (for video processing)

### Installation

1. Install ImageMagick (macOS):
```bash
brew install imagemagick
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
python app.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Generate Captions with Word Highlighting

**Endpoint:** `POST /generate-captions/`

**Parameters:**
- `video`: Video file (MP4 format)
- `language`: Language code (default: "en")
- `style`: Caption style (default: "classic")

**Available Styles:**
- `classic`: Traditional white text with black outline
- `youtube`: YouTube-style with semi-transparent background
- `minimal`: Clean, centered text with background
- `rich`: Gold text with Impact font
- `live`: Live streaming style
- `bold_white_top`: Large white text at top
- `bold_outline`: Bold text with thick outline
- `modern_karaoke`: Large text with karaoke-style highlighting
- `white_background`: Black text on solid white background for maximum readability

### 2. Generate Modern Karaoke Subtitles

**Endpoint:** `POST /generate-modern-karaoke-ffmpeg/`

**Parameters:**
- `video`: Video file (MP4 format)
- `language`: Language code (default: "en")

Creates karaoke-style subtitles with:
- Solid white background
- Black text with blue word highlighting
- Bold Arial font
- Word-by-word synchronization
- Automatic text wrapping

### 3. API Documentation

**Endpoint:** `GET /docs`

Access interactive API documentation (Swagger UI) at `http://localhost:8000/docs`

## Usage Examples

### Basic Caption Generation

```bash
curl -X POST "http://localhost:8000/generate-captions/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=white_background"
```

### Karaoke-Style Subtitles

```bash
curl -X POST "http://localhost:8000/generate-modern-karaoke-ffmpeg/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en"
```

## Technical Details

- **Whisper Model**: Uses "base" model for transcription (can be upgraded to larger models for better accuracy)
- **Video Processing**: MoviePy for video manipulation and caption embedding
- **Text Rendering**: PIL (Pillow) for custom text rendering with advanced styling
- **Font Support**: System fonts with fallback options
- **Performance**: Optimized for processing speed with efficient memory management

## Supported Formats

- **Input**: MP4 video files
- **Output**: MP4 video files with embedded captions
- **Languages**: All languages supported by OpenAI Whisper

## Notes

- The API automatically handles temporary file cleanup
- Word-level timestamps are used for precise synchronization
- Text wrapping is automatically applied to prevent overflow
- Font fallbacks are implemented for cross-platform compatibility
- The karaoke endpoint provides the most advanced word-level highlighting 