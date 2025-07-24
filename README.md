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
- **Cloud Deployment**: Hosted on Modal for production-ready API access

## 🚀 Hosted APIs

The Auto Caption Generator is now deployed on Modal and available as hosted APIs:

### Production Endpoints

**1. Generate Subtitles (Classic Style)**
```
https://garvitb--auto-caption-generator-generate-subtitles.modal.run
```
- Supports multiple subtitle styles (classic, modern, karaoke-style)
- Chunked subtitles for better readability
- Multiple language support

**2. Generate Live Subtitles (Karaoke Style)**
```
https://garvitb--auto-caption-generator-generate-live-subtitles.modal.run
```
- Real-time karaoke-style subtitles
- Word-by-word highlighting
- Dynamic text rendering

### Usage Examples

**Generate Classic Subtitles:**
```bash
curl -X POST "https://garvitb--auto-caption-generator-generate-subtitles.modal.run" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=classic"
```

**Generate Karaoke-Style Subtitles:**
```bash
curl -X POST "https://garvitb--auto-caption-generator-generate-live-subtitles.modal.run" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "video=@your_video.mp4" \
     -F "language=en"
```

> **Note:** The hosted APIs require an OpenAI API key to be configured in Modal secrets.

## Setup

### Prerequisites

- Python 3.8+
- ImageMagick (for text rendering)
- FFmpeg (for video processing)
- OpenAI API key
- AWS S3 bucket (optional, for cloud storage)

## Modal Deployment with S3

The app can be deployed to Modal with S3 upload functionality for cloud storage.

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
   modal secret create s3-credentials \
     S3_BUCKET_NAME=your_bucket_name \
     AWS_ACCESS_KEY_ID=your_access_key \
     AWS_SECRET_ACCESS_KEY=your_secret_key \
     AWS_REGION=us-east-1
   ```

   Or use the provided setup script:
   ```bash
   ./setup_s3_credentials.sh
   ```

### Deploy to Modal

1. **Automatic Deployment**: Use the provided deployment script
   ```bash
   ./deploy_to_modal.sh
   ```

2. **Manual Deployment**: Deploy directly with Modal
   ```bash
   python modal_deploy.py
   ```

### Modal Endpoints with S3

When deployed to Modal with S3 credentials configured, the endpoints will:

- **Upload processed videos to S3** automatically
- **Return JSON responses** with S3 URLs instead of file downloads
- **Clean up local files** after successful upload
- **Fall back to file downloads** if S3 upload fails

**Example Response from Modal with S3:**
```json
{
  "success": true,
  "message": "Video processed successfully",
  "video_url": "https://your-bucket.s3.us-east-1.amazonaws.com/captioned_abc123.mp4",
  "filename": "captioned_abc123.mp4",
  "style": "classic",
  "language": "en"
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

4. (Optional) Set up S3 for cloud storage:
   - Create an S3 bucket in your AWS account
   - Create an IAM user with S3 access permissions
   - Add S3 credentials to your `.env` file:
   ```
   S3_BUCKET_NAME=your_s3_bucket_name
   AWS_ACCESS_KEY_ID=your_aws_access_key_id
   AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
   AWS_REGION=us-east-1
   ```

5. Run the API server:
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

**Response Format:**
- If S3 is configured: Returns JSON with video URL
- If S3 is not configured: Returns the video file directly

**JSON Response (when S3 is configured):**
```json
{
  "success": true,
  "message": "Video processed successfully",
  "video_url": "https://your-bucket.s3.us-east-1.amazonaws.com/captioned_abc123.mp4",
  "filename": "captioned_abc123.mp4",
  "style": "classic",
  "language": "en"
}
```

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
  "type": "karaoke"
}
```

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
- **Output**: MP4 video files with embedded subtitles
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

## Font Options for Live Subtitles

The `/generate-live-subtitles/` endpoint supports the following fonts. Use the corresponding value for the `font` parameter:

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
| Circular Std        | circular_std        |

**Example usage:**
- For Montserrat: `montserrat`
- For Times New Roman: `times_new_roman`
- For Comic Sans: `comic_sans`

**To add more fonts:**
1. Place the `.ttf` file in the `fonts/` directory.
2. Update the `font_map` dictionary in `app.py` to include your new font and its parameter value. 