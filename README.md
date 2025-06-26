# Auto Caption Generator

An advanced video captioning API that automatically generates and embeds captions into videos using OpenAI's Whisper API for transcription. Features chunked subtitles, multiple styles, and modern visual options.

## Features

- **Automatic Transcription**: Uses OpenAI Whisper API for accurate speech-to-text conversion
- **Chunked Subtitles**: For 'classic' and 'centered' styles, subtitles are split into 5-6 word groups for better readability
- **Multiple Caption Styles**: Three distinct visual styles
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
- OpenAI API key

### Installation

1. Install ImageMagick (macOS):
```bash
brew install imagemagick
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

4. Run the API server:
```bash
python app.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Generate Subtitles

**Endpoint:** `POST /generate-subtitles/`

**Parameters:**
- `video`: Video file (MP4 format)
- `language`: Language code (default: "en")
- `style`: Caption style (default: "classic")

**Available Styles:**
- `youtube`: Bottom, white text, large font, semi-transparent black background, no border
- `classic`: Bottom, white text, Arial, thin black border, no background, chunked to 5-6 words
- `centered`: Center, bold white text, large font, semi-transparent black background, no border, chunked to 5-6 words

> **Note:** This endpoint uses segment-level timing (not word-level karaoke highlighting).

### 2. Generate Live Karaoke-Style Subtitles

**Endpoint:** `POST /generate-live-subtitles/`

**Parameters:**
- `video`: Video file (MP4 format)
- `language`: Language code (default: "en")

Creates karaoke-style subtitles with:
- Solid white background
- Black text with blue word highlighting
- Bold Arial font
- Word-by-word synchronization (word-level timing)
- Automatic text wrapping

### 3. API Documentation

**Endpoint:** `GET /docs`

Access interactive API documentation (Swagger UI) at `http://localhost:8000/docs`

## Usage Examples

### Basic Subtitle Generation

```bash
curl -X POST "http://localhost:8000/generate-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=classic"
```

### Centered Chunked Subtitles

```bash
curl -X POST "http://localhost:8000/generate-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=centered"
```

### YouTube-Style Subtitles

```bash
curl -X POST "http://localhost:8000/generate-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=youtube"
```

### Live Karaoke-Style Subtitles (Word-level)

```bash
curl -X POST "http://localhost:8000/generate-live-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en"
```

## Technical Details

- **Whisper API**: Uses OpenAI's Whisper API for transcription (requires internet connection and API key)
    - `/generate-subtitles/` uses the OpenAI Python client (segment-level timing)
    - `/generate-live-subtitles/` uses the HTTP API directly for word-level timing
- **Video Processing**: MoviePy for video manipulation and caption embedding
- **Text Rendering**: PIL (Pillow) for custom text rendering with advanced styling
- **Font Support**: System fonts with fallback options
- **Performance**: Optimized for processing speed with efficient memory management
- **API Costs**: Transcription costs are based on OpenAI's pricing (typically $0.006 per minute)

## Supported Formats

- **Input**: MP4 video files
- **Output**: MP4 video files with embedded captions
- **Languages**: All languages supported by OpenAI Whisper

## Notes

- The API automatically handles temporary file cleanup
- `/generate-live-subtitles/` provides word-level karaoke highlighting
- `/generate-subtitles/` provides segment-level subtitles (chunked for readability)
- Subtitles for 'classic' and 'centered' are chunked for readability
- The classic style has a thin black border for clarity
- The centered style is bold, large, and highly visible
- The YouTube style is large, bottom-aligned, and backgrounded for maximum readability
- Requires an active internet connection for OpenAI API calls
- API usage is subject to OpenAI's rate limits and pricing 