import os
from moviepy.config import change_settings

# Configure ImageMagick with the correct path
IMAGEMAGICK_BINARY = "/opt/homebrew/bin/magick"  # Confirmed correct path on your system
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})

from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
os.environ["MAGICK_HOME"] = "/opt/homebrew/opt/imagemagick"  # Update this path based on your ImageMagick installation
os.environ["PATH"] = f"{os.environ['MAGICK_HOME']}/bin:" + os.environ.get("PATH", "")
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI
import tempfile
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from pydantic import BaseModel
from typing import Optional
import uuid
import textwrap
import traceback
import srt
import datetime
import subprocess
import json
import csv
from dotenv import load_dotenv
import httpx
import asyncio
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

app = FastAPI(title="Auto Caption Generator")

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("[WARNING] OPENAI_API_KEY not found in environment variables. Please set it in your .env file or environment.")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)

# Initialize S3 client
s3_client = None
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")

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
    print("[WARNING] S3 credentials not found. Please set S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY in your .env file.")

def upload_to_s3(file_path: str, object_name: str = None) -> Optional[str]:
    """Upload a file to S3 and return the public URL"""
    if not s3_client:
        raise HTTPException(status_code=500, detail="S3 not configured")
    
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    try:
        s3_client.upload_file(file_path, s3_bucket_name, object_name)
        
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

# Function to transcribe using OpenAI Whisper API
async def transcribe_with_openai(video_file_path: str, language: str = "en"):
    """Extract audio from video and transcribe using OpenAI's Whisper API"""
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Extract audio from video
        print("[INFO] Extracting audio from video...")
        audio_path = video_file_path.replace(".mp4", "_audio.wav")
        
        # Extract audio using ffmpeg
        extract_cmd = [
            "ffmpeg", "-i", video_file_path, 
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
            compressed_audio_path = video_file_path.replace(".mp4", "_audio_compressed.wav")
            
            # Calculate target bitrate to fit within 25MB
            # Get video duration for bitrate calculation
            probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_file_path]
            duration_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout.strip())
            
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
        
        # Transcribe the audio
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json"
            )
        
        # Clean up audio files
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        compressed_audio_path = video_file_path.replace(".mp4", "_audio_compressed.wav")
        if os.path.exists(compressed_audio_path):
            os.unlink(compressed_audio_path)
        
        return response
    except Exception as e:
        print(f"[ERROR] OpenAI transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI transcription failed: {str(e)}")

# Function to transcribe using OpenAI Whisper HTTP API for word-level timestamps
async def transcribe_with_openai_http(video_file_path: str, language: str = "en"):
    """Extract audio from video and transcribe using OpenAI's Whisper API via HTTP for word-level timestamps"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Extract audio from video
        print("[INFO] Extracting audio from video...")
        audio_path = video_file_path.replace(".mp4", "_audio.wav")
        
        # Extract audio using ffmpeg
        extract_cmd = [
            "ffmpeg", "-i", video_file_path, 
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
            compressed_audio_path = video_file_path.replace(".mp4", "_audio_compressed.wav")
            
            # Calculate target bitrate to fit within 25MB
            # Get video duration for bitrate calculation
            probe_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", video_file_path]
            duration_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout.strip())
            
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
        
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}"
        }
        data = {
            "model": "whisper-1",
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word"
        }
        
        # Get file size for timeout calculation
        file_size = os.path.getsize(audio_path)
        # Estimate timeout: 30 seconds base + 1 second per MB
        timeout_seconds = 30 + (file_size / (1024 * 1024))
        timeout_seconds = min(timeout_seconds, 300)  # Max 5 minutes
        
        print(f"[INFO] File size: {file_size / (1024*1024):.1f}MB, Timeout: {timeout_seconds:.0f}s")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
                    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                        print(f"[INFO] Attempt {attempt + 1}/{max_retries}: Sending to OpenAI...")
                        response = await client.post(url, headers=headers, data=data, files=files)
                    
                    if response.status_code == 200:
                        print("[INFO] Transcription successful!")
                        result = response.json()
                        
                        # Clean up audio files
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                        compressed_audio_path = video_file_path.replace(".mp4", "_audio_compressed.wav")
                        if os.path.exists(compressed_audio_path):
                            os.unlink(compressed_audio_path)
                        
                        return result
                    else:
                        print(f"[ERROR] OpenAI HTTP API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                        if attempt == max_retries - 1:
                            raise HTTPException(status_code=500, detail=f"OpenAI HTTP API error: {response.text}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        
            except httpx.TimeoutException as e:
                print(f"[WARNING] Timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"OpenAI transcription timed out after {max_retries} attempts. File may be too large or network too slow.")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                print(f"[ERROR] OpenAI HTTP transcription failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"OpenAI HTTP transcription failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
    except Exception as e:
        print(f"[ERROR] Audio extraction or transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio extraction or transcription failed: {str(e)}")

# Update the SUBTITLE_PRESETS
SUBTITLE_PRESETS = {
    "youtube": {
        "font": "DejaVu-Sans",
        "fontsize": 48,
        "color": "white",
        "highlight_color": "#00A5FF",
        "bg_color": "rgba(0,0,0,0.7)",
        "position": ("center", "bottom"),
        "method": "label",
        "align": "center",
        "spacing": 5
    },
    "classic": {
        "font": "Arial",
        "fontsize": 48,
        "color": "white",
        "highlight_color": "#00A5FF",
        "bg_color": None,
        "position": ("center", "bottom"),
        "method": "label",
        "align": "center",
        "stroke_color": "black",
        "stroke_width": 1,
        "spacing": 5
    },
    "centered": {
        "font": "Arial-Bold",
        "fontsize": 48,
        "color": "white",
        "highlight_color": "#00A5FF",
        "bg_color": "rgba(0,0,0,0.7)",
        "position": ("center", "center"),
        "method": "label",
        "align": "center",
        "spacing": 6
    }
}

class CaptionRequest(BaseModel):
    language: str = "en"
    style: str = "classic"  # Default style

@app.post("/generate-subtitles/")
async def generate_subtitles(
    video: UploadFile = File(...),
    language: str = "en",
    style: str = "classic"
):
    try:
        print("[INFO] Starting caption generation...")
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        print(f"[INFO] Video saved to {temp_video_path}")

        # Load the video
        print("[INFO] Loading video clip...")
        video_clip = VideoFileClip(temp_video_path)
        print(f"[INFO] Video loaded: duration={video_clip.duration}, size={video_clip.size}")
        
        # Now calculate safe area and max width/height
        safe_margin_y = int(video_clip.h * 0.08)  # 8% margin top/bottom
        max_width = int(video_clip.w * 0.9)       # 90% of video width
        max_height = int(video_clip.h * 0.25)     # 25% of video height for subtitles
        BOTTOM_BUFFER = -120
        pad = 30 if style == "rich" else 0
        style_settings = SUBTITLE_PRESETS[style]
        font_size = style_settings["fontsize"]
        min_font_size = 14
        textclip_kwargs = {
            "color": style_settings["color"],
            "font": style_settings["font"],
            "method": "caption",
            "align": style_settings.get("align", "center")
        }
        if style_settings.get("bg_color") is not None:
            textclip_kwargs["bg_color"] = style_settings["bg_color"]
        if style_settings.get("stroke_color"):
            textclip_kwargs["stroke_color"] = style_settings["stroke_color"]
        if style_settings.get("stroke_width"):
            textclip_kwargs["stroke_width"] = style_settings["stroke_width"]

        def get_position(txt_clip):
            pos = style_settings["position"]
            is_bottom = pos[1] == "bottom"
            is_top = pos[1] == "top"
            is_center = pos[1] == "center"
            if is_bottom:
                y = video_clip.h - txt_clip.h + BOTTOM_BUFFER
            elif is_top:
                y = safe_margin_y
            else:  # center
                y = (video_clip.h - txt_clip.h) // 2
            return ("center", y)

        # Transcribe the video using OpenAI
        print("[INFO] Transcribing video...")
        result = await transcribe_with_openai(temp_video_path, language)
        print("[INFO] Transcription complete.")
        
        # Create output video with captions
        output_path = f"output_{uuid.uuid4()}.mp4"
        print(f"[INFO] Output path: {output_path}")
        
        print("[INFO] Generating SRT subtitles...")
        # Compose SRT using srt library
        subs = []
        for i, seg in enumerate(result.segments):
            subs.append(
                srt.Subtitle(
                    index=i+1,
                    start=datetime.timedelta(seconds=seg["start"]),
                    end=datetime.timedelta(seconds=seg["end"]),
                    content=seg["text"].strip()
                )
            )
        srt_content = srt.compose(subs)
        with open("output.srt", "w", encoding="utf-8") as f:
            f.write(srt_content)
        print("[INFO] SRT file written.")
        
        # Subtitle creation
        print("[INFO] Creating subtitle clips...")
        captions = []
        def chunk_words(text, chunk_size=6):
            words = text.split()
            return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        for seg in result.segments:
            text = seg["text"].strip()
            start = seg["start"]
            end = seg["end"]
            if style in ["classic", "centered"]:
                chunks = chunk_words(text, 6)
                duration = (end - start) / max(1, len(chunks))
                for i, chunk in enumerate(chunks):
                    chunk_start = start + i * duration
                    chunk_end = min(end, chunk_start + duration)
                    font_size_try = font_size
                    while font_size_try >= min_font_size:
                        textclip_kwargs["fontsize"] = font_size_try
                        try:
                            txt_clip = TextClip(chunk, size=(max_width - 2 * pad, None), **textclip_kwargs)
                        except Exception as e:
                            raise HTTPException(status_code=500, detail=f"TextClip error: {str(e)} | kwargs: {textclip_kwargs}")
                        if txt_clip.w <= (max_width - 2 * pad) and txt_clip.h <= max_height:
                            break
                        font_size_try -= 2
                    position = get_position(txt_clip)
                    txt_clip = txt_clip.set_position(position)
                    txt_clip = txt_clip.set_start(chunk_start).set_end(chunk_end)
                    captions.append(txt_clip)
            else:
                font_size_try = font_size
                while font_size_try >= min_font_size:
                    textclip_kwargs["fontsize"] = font_size_try
                    try:
                        txt_clip = TextClip(text, size=(max_width - 2 * pad, None), **textclip_kwargs)
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"TextClip error: {str(e)} | kwargs: {textclip_kwargs}")
                    if txt_clip.w <= (max_width - 2 * pad) and txt_clip.h <= max_height:
                        break
                    font_size_try -= 2
                position = get_position(txt_clip)
                txt_clip = txt_clip.set_position(position)
                txt_clip = txt_clip.set_start(start).set_end(end)
                captions.append(txt_clip)
        print(f"[INFO] Created {len(captions)} subtitle clips.")
        
        # Combine video with captions
        print("[INFO] Compositing video with subtitles...")
        final_video = CompositeVideoClip([video_clip] + captions)
        print("[INFO] Writing final video file...")
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print("[INFO] Video writing complete.")
        
        # Clean up temporary files
        os.unlink(temp_video_path)
        print("[INFO] Temporary video file deleted.")
        
        # Upload to S3 and return JSON response
        if s3_client:
            try:
                print("[INFO] Uploading video to S3...")
                output_filename = f"captioned_{uuid.uuid4().hex[:8]}.mp4"
                video_url = upload_to_s3(output_path, output_filename)
                
                # Clean up local file
                os.unlink(output_path)
                
                return JSONResponse(content={
                    "success": True,
                    "message": "Video processed successfully",
                    "video_url": video_url,
                    "filename": output_filename,
                    "style": style,
                    "language": language
                })
            except Exception as e:
                print(f"[ERROR] S3 upload failed: {e}")
                # Fall back to file response if S3 fails
                return FileResponse(
                    output_path,
                    media_type="video/mp4",
                    filename="captioned_video.mp4"
                )
        else:
            # If S3 is not configured, return file response
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename="captioned_video.mp4"
            )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Exception occurred: {e}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"{str(e)}\nTraceback:\n{tb}")

def create_text_clip(text, style_settings, highlighted_word=None, video_width=1920):
    """Create a text clip with proper styling and background"""
    def make_frame(t):
        # Set max text width to 80% of video width
        max_text_width = int(video_width * 0.8)
        font_size = style_settings["fontsize"]
        try:
            font = ImageFont.truetype(style_settings["font"], font_size)
        except Exception as e:
            print(f"[WARNING] Could not load font {style_settings['font']}: {e}")
            font = ImageFont.load_default()

        # Word wrapping
        words = text.split()
        lines = []
        current_line = ""
        draw_test = ImageDraw.Draw(Image.new('RGBA', (10, 10)))
        for word in words:
            test_line = (current_line + " " + word).strip()
            bbox = draw_test.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
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
            bbox = draw_test.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        total_height = sum(line_heights) + (len(lines) - 1) * 10
        max_width = max(line_widths)

        img_w = video_width
        img_h = total_height + 60  # padding
        img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Check if this is a white background style
        is_white_bg = style_settings.get("white_bg", False)
        
        if is_white_bg:
            # Draw solid white background rectangle with padding
            bg_padding = style_settings.get("bg_padding", 20)
            rect_x0 = (img_w - max_width) // 2 - bg_padding
            rect_y0 = (img_h - total_height) // 2 - bg_padding
            rect_x1 = (img_w + max_width) // 2 + bg_padding
            rect_y1 = (img_h + total_height) // 2 + bg_padding
            # Solid white background (opaque)
            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(255, 255, 255, 255))
        else:
            # Original semi-transparent background logic
            bg_margin_x = 30
            bg_margin_y = 20
            rect_x0 = (img_w - max_width) // 2 - bg_margin_x
            rect_y0 = (img_h - total_height) // 2 - bg_margin_y
            rect_x1 = (img_w + max_width) // 2 + bg_margin_x
            rect_y1 = (img_h + total_height) // 2 + bg_margin_y
            bg_opacity = int(255 * style_settings.get("bg_opacity", 0.5))
            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, bg_opacity))

        # Draw each line with stroke and highlight
        y = (img_h - total_height) // 2
        for line in lines:
            # Highlight word if present
            words_in_line = line.split()
            word_sizes = []
            for word in words_in_line:
                bbox = draw.textbbox((0, 0), word + " ", font=font)
                word_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
            total_line_width = sum(w for w, h in word_sizes) + style_settings["spacing"] * (len(words_in_line) - 1)
            x = (img_w - total_line_width) // 2
            current_x = x
            for i, (word, (w, h)) in enumerate(zip(words_in_line, word_sizes)):
                # Draw stroke (only if not white background style)
                if not is_white_bg and style_settings.get("stroke_width", 0) > 0:
                    stroke_width = style_settings["stroke_width"]
                    offsets = []
                    for dx in range(-stroke_width, stroke_width + 1):
                        for dy in range(-stroke_width, stroke_width + 1):
                            if dx != 0 or dy != 0:
                                offsets.append((dx, dy))
                    for offset_x, offset_y in offsets:
                        draw.text((current_x + offset_x, y + offset_y), word, font=font, fill=style_settings["stroke_color"])
                
                # Draw word
                color = style_settings["highlight_color"] if word == highlighted_word else style_settings["color"]
                draw.text((current_x, y), word, font=font, fill=color)
                current_x += w + style_settings["spacing"]
            y += h + 10  # line spacing

        img_array = np.array(img)
        rgb = img_array[..., :3]
        alpha = img_array[..., 3]
        mask = alpha / 255.0
        return rgb * mask[..., np.newaxis]

    duration = 0.5  # Default duration, will be overridden
    clip = VideoClip(lambda t: make_frame(t), duration=duration)
    def make_mask(t):
        frame = make_frame(t)
        return np.mean(frame, axis=2)
    clip.mask = VideoClip(make_mask, duration=duration)
    return clip

@app.post("/generate-live-subtitles/")
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
    Generate a video with live karaoke-style subtitles with advanced styling options.
    Supports 30+ fonts and custom colors, backgrounds, and styling.
    """
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        print(f"[INFO] Video saved to {temp_video_path}")

        # Transcribe using OpenAI HTTP API for word-level timestamps
        print("[INFO] Running transcription (HTTP API, word-level)...")
        result = await transcribe_with_openai_http(temp_video_path, language)
        
        # Debug: Print the structure of the response
        print(f"[DEBUG] Response keys: {list(result.keys())}")
        print(f"[DEBUG] Number of segments: {len(result.get('segments', []))}")
        if result.get('segments'):
            first_segment = result['segments'][0]
            print(f"[DEBUG] First segment keys: {list(first_segment.keys())}")
            if 'words' in first_segment:
                print(f"[DEBUG] Number of words in first segment: {len(first_segment['words'])}")
                if first_segment['words']:
                    print(f"[DEBUG] First word structure: {first_segment['words'][0]}")

        # Load the video
        video_clip = VideoFileClip(temp_video_path)
        video_width, video_height = video_clip.size
        
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
        FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
        font_map = {
            # Available fonts (only these are actually present in fonts/ directory)
            "arial": os.path.join(FONT_DIR, "Arial-Bold.ttf"),
            "georgia": os.path.join(FONT_DIR, "Georgia-Bold.ttf"),
            "montserrat": os.path.join(FONT_DIR, "Montserrat-Bold.ttf"),
            "verdana": os.path.join(FONT_DIR, "Verdana-Bold.ttf"),
            "comic_sans": os.path.join(FONT_DIR, "ComicSansMS-Bold.ttf"),
            "times_new_roman": os.path.join(FONT_DIR, "TimesNewRoman-Bold.ttf"),
            "courier_new": os.path.join(FONT_DIR, "CourierNew-Bold.ttf"),
            "trebuchet_ms": os.path.join(FONT_DIR, "TrebuchetMS-Bold.ttf"),
            "tahoma": os.path.join(FONT_DIR, "Tahoma-Bold.ttf"),
            
            # Additional popular fonts (successfully downloaded)
            "roboto": os.path.join(FONT_DIR, "Roboto-Bold.ttf"),
            "open_sans": os.path.join(FONT_DIR, "OpenSans-Bold.ttf"),
            "raleway": os.path.join(FONT_DIR, "Raleway-Bold.ttf"),
        }
        
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

        # Special handling for circular_std
        if selected_font.lower() == "circular_std":
            fontsize = 150
            
        karaoke_clips = []

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
                    
                    # Determine word color
                    if word == highlighted_word or clean_word == clean_highlighted:
                        word_color = highlight_color
                    else:
                        word_color = font_color
                    
                    # Handle font_fill_style if provided
                    if fill_style and fill_style.get("type") == "solid":
                        word_color = fill_style.get("value", word_color)
                    
                    # Convert color to RGB
                    word_rgb = color_to_rgb(word_color)
                    
                    # Draw bold text (draw twice with 1px offset)
                    draw.text((current_x, y), word, font=font_obj, fill=word_rgb)
                    draw.text((current_x+1, y), word, font=font_obj, fill=word_rgb)
                    current_x += word_width
                
                y += line_heights[idx] + 10
            
            rgb_img = Image.new('RGB', img.size, bg_rgb)
            rgb_img.paste(img, mask=img.split()[-1])
            img_array = np.array(rgb_img)
            return img_array

        # Process segments and words
        total_words = 0
        segments = result.get("segments", [])
        if segments:
            for segment in segments:
                words = segment.get("words", [])
                if not words:
                    continue
                total_words += len(words)
                print(f"[DEBUG] Processing segment with {len(words)} words")
                # Split words into chunks of 5-6
                chunk_size = 6
                chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
                for chunk in chunks:
                    chunk_texts = [w.get("word", "").upper() for w in chunk]
                    line = " ".join(chunk_texts)
                    # Get timing from the first and last word in chunk
                    if chunk and len(chunk) > 0:
                        chunk_start = chunk[0].get("start", 0)
                        chunk_end = chunk[-1].get("end", chunk_start + 1)
                        # For each word in the chunk, create a clip with the chunk text, highlighting the current word
                        for i, w in enumerate(chunk):
                            highlight_word = w.get("word", "").upper()
                            start = w.get("start", chunk_start)
                            end = w.get("end", start + 0.5)
                            print(f"[DEBUG] Rendering karaoke subtitle: '{line}' | Highlight: '{highlight_word}' | Start: {start} | End: {end}")
                            img_array = create_karaoke_image(line, highlight_word, video_width)
                            if img_array is None:
                                print("[ERROR] Failed to create subtitle image. Skipping this subtitle.")
                                continue
                            if img_array.size == 0 or len(img_array.shape) < 2:
                                print("[ERROR] Empty or invalid image generated for subtitle. Skipping this subtitle.")
                                continue
                            h, w = img_array.shape[:2]
                            txt_clip = VideoClip(lambda t, arr=img_array: arr, duration=end-start)
                            txt_clip = txt_clip.set_position(("center", video_height - h - 150))
                            txt_clip = txt_clip.set_start(start).set_end(end)
                            karaoke_clips.append(txt_clip)
        else:
            # If no segments, use top-level words
            words = result.get("words", [])
            if words:
                total_words = len(words)
                print(f"[DEBUG] Processing top-level words: {total_words}")
                chunk_size = 6
                chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
                for chunk in chunks:
                    chunk_texts = [w.get("word", "").upper() for w in chunk]
                    line = " ".join(chunk_texts)
                    if chunk and len(chunk) > 0:
                        chunk_start = chunk[0].get("start", 0)
                        chunk_end = chunk[-1].get("end", chunk_start + 1)
                        for i, w in enumerate(chunk):
                            highlight_word = w.get("word", "").upper()
                            start = w.get("start", chunk_start)
                            end = w.get("end", start + 0.5)
                            print(f"[DEBUG] Rendering karaoke subtitle: '{line}' | Highlight: '{highlight_word}' | Start: {start} | End: {end}")
                            img_array = create_karaoke_image(line, highlight_word, video_width)
                            if img_array is None:
                                print("[ERROR] Failed to create subtitle image. Skipping this subtitle.")
                                continue
                            if img_array.size == 0 or len(img_array.shape) < 2:
                                print("[ERROR] Empty or invalid image generated for subtitle. Skipping this subtitle.")
                                continue
                            h, w = img_array.shape[:2]
                            txt_clip = VideoClip(lambda t, arr=img_array: arr, duration=end-start)
                            txt_clip = txt_clip.set_position(("center", video_height - h - 150))
                            txt_clip = txt_clip.set_start(start).set_end(end)
                            karaoke_clips.append(txt_clip)

        print(f"[INFO] Total words processed: {total_words}")
        print(f"[INFO] Total karaoke clips created: {len(karaoke_clips)}")

        if not karaoke_clips:
            raise HTTPException(status_code=500, detail="No valid karaoke clips were generated. Check font availability and input video.")

        # Composite all karaoke clips onto the video
        final_video = CompositeVideoClip([video_clip] + karaoke_clips)
        output_path = f"output_modern_karaoke_{uuid.uuid4()}.mp4"
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f"[INFO] Video with karaoke subtitles written: {output_path}")

        # Clean up temporary files
        os.unlink(temp_video_path)
        print("[INFO] Temporary video file deleted.")

        # Upload to S3 and return JSON response
        if s3_client:
            try:
                print("[INFO] Uploading karaoke video to S3...")
                output_filename = f"karaoke_{uuid.uuid4().hex[:8]}.mp4"
                video_url = upload_to_s3(output_path, output_filename)
                
                # Clean up local file
                os.unlink(output_path)
                
                return JSONResponse(content={
                    "success": True,
                    "message": "Karaoke video processed successfully",
                    "video_url": video_url,
                    "filename": output_filename,
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
                    output_path,
                    media_type="video/mp4",
                    filename="modern_karaoke.mp4"
                )
        else:
            # If S3 is not configured, return file response
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename="modern_karaoke.mp4"
            )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Exception occurred: {e}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=f"{str(e)}\nTraceback:\n{tb}")

@app.get("/")
async def read_root():
    return {"message": "Welcome to Auto Caption Generator API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 