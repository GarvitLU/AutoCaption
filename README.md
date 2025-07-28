# Auto Caption Generator

An advanced video captioning API that automatically generates and embeds captions into videos using OpenAI's Whisper API for transcription. Features karaoke-style live subtitles, multiple fonts, advanced styling options, and cloud storage integration.

## Features

- **Automatic Transcription**: Uses OpenAI Whisper API for accurate speech-to-text conversion
- **Live Karaoke Subtitles**: Word-by-word highlighting with real-time synchronization
- **Advanced Styling**: Custom colors, fonts, sizes, and background options
- **12 Font Support**: Arial, Georgia, Montserrat, Verdana, Comic Sans, Times New Roman, Courier New, Trebuchet MS, Tahoma, Roboto, Open Sans, Raleway
- **Multi-language Support**: Supports transcription in multiple languages
- **Cloud Storage**: Automatic S3 upload with proper video content types
- **FastAPI-based REST API**: Modern, fast API with automatic documentation
- **Video Processing**: Direct video file processing with embedded captions
- **Temporary File Management**: Automatic cleanup of temporary files
- **Modal Deployment**: Production-ready cloud deployment

## 🚀 Hosted APIs

The Auto Caption Generator is deployed on Modal and available as hosted APIs:

### Production Endpoints

**Generate Live Subtitles (Karaoke Style)**
```
https://lu-labs--auto-caption-generator-v2-generate-live-subtitles.modal.run
```
- Real-time karaoke-style subtitles with word-by-word highlighting
- 12 different fonts available
- Advanced styling options (colors, sizes, backgrounds)
- Dynamic text rendering with custom styling

### Usage Examples

**Generate Karaoke-Style Subtitles with Custom Styling:**
```bash
curl -X POST "https://lu-labs--auto-caption-generator-v2-generate-live-subtitles.modal.run" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "font_family=roboto" \
     -F "font_weight=bold" \
     -F "font_size=32" \
     -F "bg_color=#f2f2f2" \
     -F "font_color=#000000" \
     -F "highlight_color=#ffcc00"
```

## Setup

### Prerequisites

- Python 3.8+
- ImageMagick (for text rendering)
- FFmpeg (for video processing)
- OpenAI API key
- AWS S3 bucket (optional, for cloud storage)

## Modal Deployment with S3

The app is deployed to Modal with S3 upload functionality for cloud storage.

### Prerequisites for Modal Deployment

1. **Modal CLI**: Install and authenticate with Modal
   ```bash
   pip install modal
   modal token new
   ```

2. **OpenAI API Key Secret**: Create a Modal secret for your OpenAI API key
   ```bash
   modal secret create openai-api-key OPENAI_API_KEY=your_actual_api_key_here
   ```

3. **S3 Credentials Secret**: Create a Modal secret for your S3 credentials
   ```bash
   modal secret create s3-bucket-config \
     S3_BUCKET_NAME=your_bucket_name \
     AWS_ACCESS_KEY_ID=your_access_key \
     AWS_SECRET_ACCESS_KEY=your_secret_key \
     AWS_REGION=us-east-1
   ```

### Deploy to Modal

1. **Automatic Deployment**: Use the provided deployment script
   ```bash
   ./deploy_to_modal.sh
   ```

2. **Manual Deployment**: Deploy directly with Modal
   ```bash
   modal deploy modal_app.py
   ```

### Modal Endpoints with S3

When deployed to Modal with S3 credentials configured, the endpoints will:

- **Upload processed videos to S3** automatically with proper content types
- **Return JSON responses** with S3 URLs instead of file downloads
- **Clean up local files** after successful upload
- **Fall back to file downloads** if S3 upload fails

**Example Response from Modal with S3:**
```json
{
  "success": true,
  "message": "Karaoke video processed successfully",
  "video_url": "https://your-bucket.s3.us-east-1.amazonaws.com/karaoke_abc123.mp4",
  "filename": "karaoke_abc123.mp4",
  "language": "en",
  "type": "karaoke",
  "style_used": {
    "font_family": "roboto",
    "font_weight": "bold",
    "font_size": 32,
    "bg_color": "#f2f2f2",
    "font_color": "#000000",
    "highlight_color": "#ffcc00"
  }
}
```

### S3 Bucket Requirements

- **Public Read Access**: The bucket should allow public read access for the uploaded videos
- **CORS Configuration**: Configure CORS if accessing from web browsers
- **IAM Permissions**: The IAM user should have `s3:PutObject` and `s3:GetObject` permissions

**Example S3 Bucket Policy for Public Read:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
```

## 🚀 Quick Start

### Local Development

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables:**
   Create a `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   S3_BUCKET_NAME=your_s3_bucket_name
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_REGION=us-east-1
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```

4. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/
   - Generate Live Subtitles: http://localhost:8000/generate-live-subtitles/

### Local Usage Examples

**Generate Live Karaoke-Style Subtitles:**
```bash
curl -X POST "http://localhost:8000/generate-live-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en"
```

**Generate Live Karaoke-Style Subtitles with Custom Styling:**
```bash
curl -X POST "http://localhost:8000/generate-live-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "font_family=roboto" \
     -F "font_weight=bold" \
     -F "font_size=32" \
     -F "bg_color=#f2f2f2" \
     -F "font_color=#000000" \
     -F "highlight_color=#ffcc00"
```

## API Endpoints

### Generate Live Karaoke-Style Subtitles

**Endpoint:** `POST /generate-live-subtitles/`

**Parameters:**
- `video`: Video file (multipart/form-data)
- `language`: Language code (default: "en")
- `bg_color`: Background color (default: "#ffffff")
- `font_color`: Text color (default: "#000000")
- `highlight_color`: Highlight color (default: "#00A5FF")
- `font_fill_style`: JSON string for advanced styling (optional)
- `font_family`: Font selection (default: "arial")
- `font_weight`: Font weight (default: "bold")
- `font_size`: Font size in pixels (default: 48)

Creates karaoke-style subtitles with:
- Word-by-word synchronization (word-level timing)
- Customizable fonts, colors, and sizes
- Advanced styling options
- Automatic text wrapping

**Response Format:**
- If S3 is configured: Returns JSON with video URL
- If S3 is not configured: Returns the video file directly

**JSON Response (when S3 is configured):**
```json
{
  "success": true,
  "message": "Karaoke video processed successfully",
  "video_url": "https://your-bucket.s3.us-east-1.amazonaws.com/karaoke_abc123.mp4",
  "filename": "karaoke_abc123.mp4",
  "language": "en",
  "type": "karaoke",
  "style_used": {
    "font_family": "roboto",
    "font_weight": "bold",
    "font_size": 32,
    "bg_color": "#f2f2f2",
    "font_color": "#000000",
    "highlight_color": "#ffcc00"
  }
}
```

### API Documentation

**Endpoint:** `GET /docs`

Access interactive API documentation (Swagger UI) at `http://localhost:8000/docs`

## Font Options for Live Subtitles

The `/generate-live-subtitles/` endpoint supports the following fonts. Use the corresponding value for the `font_family` parameter:

| Font Name           | Parameter Value      |
|---------------------|---------------------|
| Arial               | arial               |
| Georgia             | georgia             |
| Montserrat          | montserrat          |
| Verdana             | verdana             |
| Comic Sans MS       | comic_sans          |
| Times New Roman     | times_new_roman     |
| Courier New         | courier_new         |
| Trebuchet MS        | trebuchet_ms        |
| Tahoma              | tahoma              |
| Roboto              | roboto              |
| Open Sans           | open_sans           |
| Raleway             | raleway             |

**Example usage:**
- For Montserrat: `montserrat`
- For Times New Roman: `times_new_roman`
- For Comic Sans: `comic_sans`
- For Roboto: `roboto`
- For Open Sans: `open_sans`
- For Raleway: `raleway`

**Note:** These 12 fonts are currently available and ready to use.

## Advanced Styling Options

The `/generate-live-subtitles/` endpoint supports advanced styling parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `bg_color` | string | Background color (hex format) | `#ffffff` |
| `font_color` | string | Default text color (hex format) | `#000000` |
| `highlight_color` | string | Highlight color for current word (hex format) | `#00A5FF` |
| `font_fill_style` | string | JSON string for advanced fill styling | `null` |
| `font_family` | string | Font family name | `arial` |
| `font_weight` | string | Font weight (bold, regular) | `bold` |
| `font_size` | string | Font size in pixels | `48` |

**Example API request with advanced styling:**
```bash
curl -X POST "https://lu-labs--auto-caption-generator-v2-generate-live-subtitles.modal.run" \
  -F "video=@your_video.mp4" \
  -F "language=en" \
  -F "font_family=roboto" \
  -F "font_weight=bold" \
  -F "font_size=32" \
  -F "bg_color=#f2f2f2" \
  -F "font_color=#000000" \
  -F "highlight_color=#ffcc00"
```

**Basic API Call (Backward Compatible):**
```bash
curl -X POST "https://lu-labs--auto-caption-generator-v2-generate-live-subtitles.modal.run" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "font_family=roboto"
```

## Usage Examples

### Live Karaoke-Style Subtitles (Word-level)

```bash
curl -X POST "http://localhost:8000/generate-live-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en"
```

### Live Karaoke-Style Subtitles with Custom Styling

```bash
curl -X POST "http://localhost:8000/generate-live-subtitles/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "font_family=roboto" \
     -F "font_weight=bold" \
     -F "font_size=32" \
     -F "bg_color=#f2f2f2" \
     -F "font_color=#000000" \
     -F "highlight_color=#ffcc00"
```

## Technical Details

- **Whisper API**: Uses OpenAI's Whisper API for transcription (requires internet connection and API key)
    - `/generate-subtitles/` uses the OpenAI Python client (segment-level timing)
    - `/generate-live-subtitles/` uses the HTTP API directly for word-level timing
- **Video Processing**: MoviePy for video manipulation and caption embedding
- **Text Rendering**: PIL (Pillow) for custom text rendering with advanced styling
- **Font Support**: 12 system fonts with fallback options
- **S3 Integration**: Automatic upload with proper content types for video playback
- **Performance**: Optimized for processing speed with efficient memory management
- **API Costs**: Transcription costs are based on OpenAI's pricing (typically $0.006 per minute)

## Supported Formats

- **Input**: MP4 video files
- **Output**: MP4 video files with embedded subtitles (proper content type for browser playback)
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
- S3 uploads include proper content types for browser video playback 