# Modal Deployment Guide for Auto Caption Generator

This guide will help you deploy the Auto Caption Generator to Modal, a serverless platform for Python applications.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Python 3.8+**: Ensure you have Python installed locally

## Setup Instructions

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal token new
```

Follow the prompts to authenticate with your Modal account.

### 3. Set up OpenAI API Key Secret

Create a secret in Modal to store your OpenAI API key:

```bash
modal secret create openai-api-key OPENAI_API_KEY=your_actual_api_key_here
```

Replace `your_actual_api_key_here` with your actual OpenAI API key.

### 4. Deploy the Application

Run the deployment command:

```bash
python modal_deploy.py
```

This will:
- Build the Docker image with all dependencies
- Deploy the functions to Modal
- Provide you with the endpoint URLs

## API Endpoints

Once deployed, you'll have access to the following endpoints:

### 1. Generate Subtitles
- **URL**: `https://your-app-name--generate-subtitles.modal.run`
- **Method**: POST
- **Parameters**:
  - `video`: Video file (MP4 format)
  - `language`: Language code (default: "en")
  - `style`: Caption style (default: "classic")

### 2. Generate Live Karaoke Subtitles
- **URL**: `https://your-app-name--generate-live-subtitles.modal.run`
- **Method**: POST
- **Parameters**:
  - `video`: Video file (MP4 format)
  - `language`: Language code (default: "en")

### 3. Health Check
- **URL**: `https://your-app-name--health-check.modal.run`
- **Method**: GET

## Usage Examples

### Using curl

```bash
# Generate classic subtitles
curl -X POST "https://your-app-name--generate-subtitles.modal.run" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=classic" \
     --output captioned_video.mp4

# Generate YouTube-style subtitles
curl -X POST "https://your-app-name--generate-subtitles.modal.run" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     -F "style=youtube" \
     --output youtube_captioned.mp4

# Generate karaoke-style subtitles
curl -X POST "https://your-app-name--generate-live-subtitles.modal.run" \
     -F "video=@your_video.mp4" \
     -F "language=en" \
     --output karaoke_video.mp4
```

### Using Python

```python
import requests

# Generate subtitles
with open('your_video.mp4', 'rb') as f:
    files = {'video': f}
    data = {'language': 'en', 'style': 'classic'}
    
    response = requests.post(
        'https://your-app-name--generate-subtitles.modal.run',
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        with open('output.mp4', 'wb') as f:
            f.write(response.content)
```

## Available Styles

1. **classic**: White text with black border, bottom-aligned, chunked into 5-6 words
2. **youtube**: Large white text with semi-transparent background, bottom-aligned
3. **centered**: Bold white text with background, center-aligned, chunked into 5-6 words

## Configuration

### Resource Allocation

The deployment uses the following resources:
- **Memory**: 4GB RAM
- **CPU**: 2 cores
- **Timeout**: 10 minutes per request
- **Storage**: Modal volume for temporary file storage

### Environment Variables

The following environment variables are automatically configured:
- `OPENAI_API_KEY`: Set via Modal secret
- `MAGICK_HOME`: Set to `/usr` for ImageMagick
- `PATH`: Updated to include ImageMagick binaries

## Monitoring and Logs

### View Logs

```bash
modal logs
```

### Monitor Usage

Visit the Modal dashboard at [modal.com](https://modal.com) to:
- Monitor function invocations
- View resource usage
- Check error rates
- Manage secrets

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Large videos may take longer than 10 minutes to process
   - Solution: Consider splitting large videos or increasing timeout in the code

2. **Memory Errors**: Very high-resolution videos may exceed memory limits
   - Solution: Reduce video resolution before uploading

3. **OpenAI API Errors**: Check your API key and quota
   - Solution: Verify your OpenAI API key and check usage limits

4. **Font Issues**: If text rendering fails
   - Solution: The deployment includes DejaVu fonts as fallback

### Debug Mode

To run locally for debugging:

```bash
python modal_deploy.py
```

This will start the Modal development server locally.

## Cost Optimization

- **Cold Starts**: Functions may have cold start delays on first invocation
- **Concurrent Requests**: Modal automatically scales based on demand
- **Storage**: Temporary files are automatically cleaned up
- **API Costs**: OpenAI transcription costs apply (typically $0.006 per minute)

## Security

- API keys are stored securely in Modal secrets
- Temporary files are automatically cleaned up
- No persistent storage of uploaded videos
- HTTPS endpoints are automatically provided

## Support

For issues with:
- **Modal Platform**: Check [Modal documentation](https://modal.com/docs)
- **OpenAI API**: Check [OpenAI documentation](https://platform.openai.com/docs)
- **Application Logic**: Review the code in `modal_deploy.py` 