# Force Modal rebuild - minimal test for openai==1.2.4 and pydantic==1.10.13
import os
from moviepy.config import change_settings
IMAGEMAGICK_BINARY = "/usr/bin/convert"
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})
import modal
from pathlib import Path
from fastapi import UploadFile, File, Form
from typing import Optional

# Create a Modal app
app = modal.App("auto-caption-generator-v2")

# Import necessary modules
import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

# Define the image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "python-multipart==0.0.6",
    "moviepy==1.0.3",
    "Pillow==10.0.1",
    "numpy==1.24.3",
    "openai==1.3.7",
    "httpx==0.25.2",
    "boto3==1.34.0",
    "pydantic==1.10.13",
    "srt==3.5.2"
).apt_install(
    "ffmpeg",
    "imagemagick",
    "fonts-dejavu-core"
).run_commands(
    # Configure ImageMagick to allow text operations
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"PDF\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"PDF\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"LABEL\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"LABEL\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"TEXT\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"TEXT\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"PS\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"PS\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"EPS\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"EPS\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"XPS\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"XPS\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"MVG\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"MVG\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"SVG\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"SVG\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"XML\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"XML\"\/> -->/g' /etc/ImageMagick-6/policy.xml"
)

# Create a volume for persistent storage
volume = modal.Volume.from_name("auto-caption-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    memory=4096,
    cpu=2.0,
    secrets=[
        modal.Secret.from_name("openai-api-key"),
        modal.Secret.from_name("s3-bucket-config")
    ]
)
@modal.fastapi_endpoint(method="POST")
async def generate_live_subtitles(
    video: UploadFile = File(...),
    language: str = Form("en"),
    bg_color: Optional[str] = Form(None),
    font_color: Optional[str] = Form(None),
    highlight_color: Optional[str] = Form(None),
    font_fill_style: Optional[str] = Form(None),
    font_family: Optional[str] = Form("arial"),
    font_weight: Optional[str] = Form("bold"),
    font_size: Optional[str] = Form(None)
):
    """
    Generate live karaoke-style subtitles with advanced styling options.
    Supports 12 fonts and custom colors, backgrounds, and styling.
    """
    import os
    import tempfile
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, VideoClip
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import httpx
    import asyncio
    import json
    import uuid
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    
    # Set up environment
    os.environ["MAGICK_HOME"] = "/usr"
    os.environ["PATH"] = f"{os.environ['MAGICK_HOME']}/bin:" + os.environ.get("PATH", "")
    
    # Initialize S3 client
    import boto3
    from botocore.exceptions import ClientError
    from fastapi.responses import JSONResponse
    from typing import Optional
    
    s3_client = None
    s3_bucket_name = os.getenv("S3_BUCKET_NAME")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    # Debug: Print available environment variables
    print(f"[DEBUG] Available env vars: {list(os.environ.keys())}")
    print(f"[DEBUG] S3_BUCKET_NAME: {s3_bucket_name}")
    print(f"[DEBUG] AWS_ACCESS_KEY_ID: {'SET' if aws_access_key_id else 'NOT SET'}")
    print(f"[DEBUG] AWS_SECRET_ACCESS_KEY: {'SET' if aws_secret_access_key else 'NOT SET'}")
    print(f"[DEBUG] AWS_REGION: {aws_region}")

    if s3_bucket_name and aws_access_key_id and aws_secret_access_key:
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region
            )
            print(f"[INFO] S3 client initialized for bucket: {s3_bucket_name}")
        except Exception as e:
            print(f"[WARNING] Failed to initialize S3 client: {e}")
            s3_client = None
    else:
        print("[WARNING] S3 credentials not found. Videos will be returned as file downloads.")

    def upload_to_s3(file_path: str, object_name: str = None) -> Optional[str]:
        """Upload a file to S3 and return the public URL"""
        if not s3_client:
            raise HTTPException(status_code=500, detail="S3 not configured")
        
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        try:
            # Set proper content type for video files
            extra_args = {}
            if file_path.endswith('.mp4'):
                extra_args['ContentType'] = 'video/mp4'
            elif file_path.endswith('.avi'):
                extra_args['ContentType'] = 'video/x-msvideo'
            elif file_path.endswith('.mov'):
                extra_args['ContentType'] = 'video/quicktime'
            
            s3_client.upload_file(file_path, s3_bucket_name, object_name, ExtraArgs=extra_args)
            
            # Generate public URL
            url = f"https://{s3_bucket_name}.s3.{aws_region}.amazonaws.com/{object_name}"
            print(f"[INFO] File uploaded to S3: {url}")
            return url
        except ClientError as e:
            print(f"[ERROR] S3 upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
        except Exception as e:
            print(f"[ERROR] Unexpected error during S3 upload: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    try:
        print("[INFO] Starting live subtitle generation...")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, dir="/data", suffix=".mp4") as temp_video:
            temp_video.write(video.file.read())
            temp_video_path = temp_video.name
        
        # Load the video
        video_clip = VideoFileClip(temp_video_path)
        
        # Extract and compress audio for transcription
        print("[INFO] Extracting audio for transcription...")
        audio_path = temp_video_path.replace(".mp4", "_audio.wav")
        
        # Extract audio using ffmpeg with compression
        import subprocess
        
        # First, extract audio as WAV
        extract_cmd = [
            "ffmpeg", "-i", temp_video_path, 
            "-vn", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", 
            "-y", audio_path
        ]
        subprocess.run(extract_cmd, check=True, capture_output=True)
        
        # Check file size and compress if needed
        audio_size = os.path.getsize(audio_path)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        
        if audio_size > max_size:
            print(f"[INFO] Audio file too large ({audio_size / 1024 / 1024:.2f}MB), compressing...")
            compressed_audio_path = temp_video_path.replace(".mp4", "_audio_compressed.wav")
            
            # Calculate target bitrate to fit within 25MB
            # Estimate duration from video
            duration = video_clip.duration
            target_size = max_size * 0.95  # Leave 5% buffer
            target_bitrate = int((target_size * 8) / duration)  # bits per second
            
            compress_cmd = [
                "ffmpeg", "-i", audio_path,
                "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-b:a", str(target_bitrate),
                "-y", compressed_audio_path
            ]
            subprocess.run(compress_cmd, check=True, capture_output=True)
            
            # Use compressed audio
            audio_path = compressed_audio_path
            print(f"[INFO] Audio compressed to {os.path.getsize(audio_path) / 1024 / 1024:.2f}MB")
        
        # Transcribe with word-level timestamps
        print("[INFO] Starting transcription with word-level timestamps...")
        openai_api_key = os.environ["OPENAI_API_KEY"]
        
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        data = {
            "model": "whisper-1",
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word"
        }
        
        file_size = os.path.getsize(audio_path)
        timeout_seconds = min(30 + (file_size / (1024 * 1024)), 300)
        
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, headers=headers, data=data, files=files)
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {response.text}")
        
        transcription_data = response.json()
        words = transcription_data.get("words", [])
        
        if not words:
            raise HTTPException(status_code=400, detail="No speech detected in video")
        
        # Create karaoke-style subtitle clips
        subtitle_clips = []
        
        # Font selection logic with 12 available fonts
        FONT_DIR = "/usr/share/fonts/truetype/dejavu"  # Use system fonts in Modal
        font_map = {
            # Available fonts (using system fonts in Modal)
            "arial": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "georgia": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "montserrat": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "verdana": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "comic_sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "times_new_roman": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "courier_new": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "trebuchet_ms": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "tahoma": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "roboto": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "open_sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "raleway": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        }

        # Parse style options
        fill_style = None
        if font_fill_style:
            try:
                fill_style = json.loads(font_fill_style)
            except json.JSONDecodeError:
                print("[WARNING] Invalid font_fill_style JSON, using defaults")

        # Set default values
        bg_color = bg_color or "#ffffff"
        font_color = font_color or "#000000"
        highlight_color = highlight_color or "#00A5FF"
        padding = 30  # Increased padding for better visibility
        
        # Debug color values
        print(f"[DEBUG] Colors - bg: {bg_color}, font: {font_color}, highlight: {highlight_color}")
        
        # Font selection logic
        selected_font = font_family or "arial"
        selected_weight = font_weight or "bold"
        font_key = f"{selected_font.lower()}_{selected_weight.lower()}"

        # Try to find the font with weight, fallback to just font name, then arial
        font_path = font_map.get(font_key, font_map.get(selected_font.lower(), font_map["arial"]))

        # Font size processing
        if font_size:
            try:
                fontsize = int(font_size.replace("px", ""))
            except ValueError:
                fontsize = 48
        else:
            fontsize = 48

        def create_karaoke_image(text, highlighted_word, video_width):
            print(f"[FONT DEBUG] Requested font: {selected_font}, Weight: {selected_weight}, Path: {font_path}")
            try:
                font_obj = ImageFont.truetype(font_path, fontsize)
                print(f"[FONT DEBUG] Successfully loaded font: {font_path}, fontsize: {fontsize}")
            except Exception as e:
                print(f"[ERROR] Could not load font {font_path}: {e}")
                font_obj = ImageFont.load_default()
                print(f"[FONT DEBUG] Fallback to default font.")
            
            text = text.strip().upper()
            highlighted_word = highlighted_word.strip().upper()
            
            # Word wrapping: restrict to 90% of video width
            max_text_width = int(video_width * 0.9)
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = (current_line + " " + word).strip()
                bbox = font_obj.getbbox(test_line)
                w = bbox[2] - bbox[0]
                if w > max_text_width and current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            # Calculate total text block size
            line_heights = []
            line_widths = []
            for line in lines:
                bbox = font_obj.getbbox(line)
                line_widths.append(bbox[2] - bbox[0])
                line_heights.append(bbox[3] - bbox[1])
            total_height = sum(line_heights) + (len(lines) - 1) * 10
            max_width = max(line_widths)
            
            img_width = max_width + 2 * padding
            img_height = total_height + 2 * padding
            
            # Helper function to convert color to RGB
            def color_to_rgb(color_str):
                """Convert color string to RGB tuple"""
                try:
                    # Handle hex colors
                    if color_str.startswith("#") and len(color_str) == 7:
                        return tuple(int(color_str[i:i+2], 16) for i in (1, 3, 5))
                    # Handle named colors
                    elif color_str.lower() in ['black', 'white', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']:
                        color_map = {
                            'black': (0, 0, 0),
                            'white': (255, 255, 255),
                            'red': (255, 0, 0),
                            'green': (0, 255, 0),
                            'blue': (0, 0, 255),
                            'yellow': (255, 255, 0),
                            'cyan': (0, 255, 255),
                            'magenta': (255, 0, 255)
                        }
                        return color_map[color_str.lower()]
                    else:
                        print(f"[WARNING] Invalid color format: {color_str}, using black")
                        return (0, 0, 0)
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Color parsing error for '{color_str}': {e}, using black")
                    return (0, 0, 0)
            
            # Create background with custom bg_color
            bg_rgb = color_to_rgb(bg_color)
            print(f"[DEBUG] Background RGB: {bg_rgb}")
                
            img = Image.new('RGBA', (img_width, img_height), (*bg_rgb, 255))
            draw = ImageDraw.Draw(img)
            
            y = padding
            for idx, line in enumerate(lines):
                line_words = line.split()
                line_bbox = font_obj.getbbox(line)
                line_width = line_bbox[2] - line_bbox[0]
                x = (img_width - line_width) // 2
                current_x = x
                
                for word in line_words:
                    clean_word = word.strip('.,!?;:')
                    clean_highlighted = highlighted_word.strip('.,!?;:')
                    
                    word_bbox = font_obj.getbbox(word + " ")
                    word_width = word_bbox[2] - word_bbox[0]
                    
                    # Determine word color based on highlighting
                    if word == highlighted_word or clean_word == clean_highlighted:
                        word_color = color_to_rgb(highlight_color)
                    else:
                        word_color = color_to_rgb(font_color)
                    
                    # Draw bold text (draw twice with 1px offset)
                    draw.text((current_x, y), word, font=font_obj, fill=word_color)
                    draw.text((current_x+1, y), word, font=font_obj, fill=word_color)
                    current_x += word_width
                y += line_heights[idx] + 10
            
            # Center the bar horizontally in the video
            def make_frame(t):
                rgb_img = Image.new('RGB', (video_width, img_height), (*bg_rgb, 255))
                x_offset = (video_width - img_width) // 2
                rgb_img.paste(img, (x_offset, 0), mask=img.split()[-1])
                return np.array(rgb_img)
            from moviepy.video.VideoClip import VideoClip
            return VideoClip(make_frame, duration=1)

        # --- CHUNKING AND CLIP CREATION LOGIC ---
        # Use the same chunking as your reference: 5-6 words per chunk
        def chunk_words(words, chunk_size=6):
            return [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
        # Group words into chunks and create karaoke clips
        all_words = words  # from transcription
        chunk_size = 6
        chunks = chunk_words(all_words, chunk_size)
        for chunk in chunks:
            chunk_text = " ".join([w["word"].upper() for w in chunk])
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]
            for i, w in enumerate(chunk):
                highlight_word = w["word"].upper()
                start = w["start"]
                end = w["end"]
                duration = end - start
                text_clip = create_karaoke_image(chunk_text, highlight_word, video_clip.w)
                # Bottom align: position bar at bottom with 5% margin
                bar_y = video_clip.h - 110 - int(video_clip.h * 0.05)
                text_clip = text_clip.set_position(("center", bar_y)).set_start(start).set_end(end)
                subtitle_clips.append(text_clip)
        
        # Composite video with subtitles
        print("[INFO] Compositing video with karaoke subtitles...")
        final_video = CompositeVideoClip([video_clip] + subtitle_clips)
        
        # Generate output filename
        output_filename = f"karaoke_{uuid.uuid4().hex[:8]}.mp4"
        output_path = f"/data/{output_filename}"
        
        # Write final video
        print("[INFO] Writing final video...")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        # Clean up
        video_clip.close()
        final_video.close()
        os.unlink(temp_video_path)
        
        # Clean up audio files
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        compressed_audio_path = temp_video_path.replace(".mp4", "_audio_compressed.wav")
        if os.path.exists(compressed_audio_path):
            os.unlink(compressed_audio_path)
        
        # Upload to S3 and return JSON response
        if s3_client:
            try:
                print("[INFO] Uploading karaoke video to S3...")
                s3_filename = f"karaoke_{uuid.uuid4().hex[:8]}.mp4"
                video_url = upload_to_s3(output_path, s3_filename)
                
                # Clean up local file
                os.unlink(output_path)
                
                return JSONResponse(content={
                    "success": True,
                    "message": "Karaoke video processed successfully",
                    "video_url": video_url,
                    "filename": s3_filename,
                    "language": language,
                    "type": "karaoke",
                    "style_used": {
                        "font_family": selected_font,
                        "font_weight": selected_weight,
                        "font_size": fontsize,
                        "bg_color": bg_color,
                        "font_color": font_color,
                        "highlight_color": highlight_color
                    }
                })
            except Exception as e:
                print(f"[ERROR] S3 upload failed: {e}")
                # Fall back to file response if S3 fails
                return FileResponse(
                    path=output_path,
                    filename=output_filename,
                    media_type="video/mp4"
                )
        else:
            # If S3 is not configured, return file response
            return FileResponse(
                path=output_path,
                filename=output_filename,
                media_type="video/mp4"
            )
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.function(image=image)
@modal.fastapi_endpoint()
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Auto Caption Generator is running"}

# For local development
if __name__ == "__main__":
    app.deploy() 