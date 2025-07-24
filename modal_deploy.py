import modal
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

# Create a Modal app
app = modal.App("auto-caption-generator")

# Define the image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    "python-multipart==0.0.6",
    "moviepy==1.0.3",
    "pydantic==2.4.2",
    "python-dotenv==1.0.0",
    "openai==1.2.4",
    "httpx",
    "srt",
    "Pillow",
    "numpy"
).apt_install(
    "ffmpeg",
    "imagemagick",
    "fonts-dejavu-core",
    "fonts-liberation"
).run_commands(
    # Configure ImageMagick policy to allow reading/writing
    "sed -i 's/rights=\"none\" pattern=\"PDF\"/rights=\"read|write\" pattern=\"PDF\"/' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/rights=\"none\" pattern=\"LABEL\"/rights=\"read|write\" pattern=\"LABEL\"/' /etc/ImageMagick-6/policy.xml"
)

# Create a volume for persistent storage
volume = modal.Volume.from_name("caption-generator-volume", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,  # 10 minutes timeout
    memory=4096,  # 4GB RAM
    cpu=2.0,      # 2 CPU cores
    secrets=[modal.Secret.from_name("openai-api-key")]
)
@modal.fastapi_endpoint(method="POST")
async def generate_subtitles(
    video: UploadFile = File(...),
    language: str = Form("en"),
    style: str = Form("classic")
):
    """Generate subtitles for uploaded video"""
    import os
    import tempfile
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from openai import OpenAI
    import uuid
    import textwrap
    
    # Set up environment
    os.environ["MAGICK_HOME"] = "/usr"
    os.environ["PATH"] = f"{os.environ['MAGICK_HOME']}/bin:" + os.environ.get("PATH", "")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Subtitle presets
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
    
    try:
        print("[INFO] Starting caption generation...")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, dir="/data", suffix=".mp4") as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        
        print(f"[INFO] Video saved to {temp_video_path}")

        # Load the video
        print("[INFO] Loading video clip...")
        video_clip = VideoFileClip(temp_video_path)
        print(f"[INFO] Video loaded: duration={video_clip.duration}, size={video_clip.size}")
        
        # Calculate safe area and max width/height
        safe_margin_y = int(video_clip.h * 0.08)
        max_width = int(video_clip.w * 0.9)
        max_height = int(video_clip.h * 0.25)
        BOTTOM_BUFFER = -120
        pad = 30 if style == "rich" else 0
        style_settings = SUBTITLE_PRESETS[style]
        font_size = style_settings["fontsize"]
        min_font_size = 14
        
        # Transcribe audio
        print("[INFO] Starting transcription...")
        with open(temp_video_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=language,
                response_format="verbose_json"
            )
        
        print("[INFO] Transcription completed")
        
        # Process transcription results
        segments = response.segments
        if not segments:
            raise HTTPException(status_code=400, detail="No speech detected in video")
        
        # Create subtitle clips
        subtitle_clips = []
        
        def chunk_words(text, chunk_size=6):
            """Split text into chunks of specified size"""
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
            return chunks
        
        def create_text_clip(text, style_settings, highlighted_word=None, video_width=1920):
            """Create a text clip with the specified style"""
            def make_frame(t):
                # Create image with text
                img = Image.new('RGBA', (video_width, 200), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                # Try to use a system font
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
                except:
                    font = ImageFont.load_default()
                
                # Calculate text position
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (video_width - text_width) // 2
                y = (200 - text_height) // 2
                
                # Draw background if specified
                if style_settings.get("bg_color"):
                    padding = 20
                    bg_bbox = (x - padding, y - padding, x + text_width + padding, y + text_height + padding)
                    draw.rectangle(bg_bbox, fill=style_settings["bg_color"])
                
                # Draw text
                draw.text((x, y), text, fill=style_settings["color"], font=font)
                
                return np.array(img)
            
            return TextClip(make_frame, duration=1, size=(video_width, 200))
        
        # Process segments and create subtitle clips
        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            text = segment.text.strip()
            
            if style in ["classic", "centered"]:
                # Chunk the text for better readability
                chunks = chunk_words(text, 6)
                chunk_duration = (end_time - start_time) / len(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_start = start_time + (i * chunk_duration)
                    chunk_end = chunk_start + chunk_duration
                    
                    text_clip = create_text_clip(chunk, style_settings, video_width=video_clip.w)
                    text_clip = text_clip.set_position(style_settings["position"]).set_duration(chunk_duration)
                    text_clip = text_clip.set_start(chunk_start)
                    subtitle_clips.append(text_clip)
            else:
                # Use full text for other styles
                text_clip = create_text_clip(text, style_settings, video_width=video_clip.w)
                text_clip = text_clip.set_position(style_settings["position"]).set_duration(end_time - start_time)
                text_clip = text_clip.set_start(start_time)
                subtitle_clips.append(text_clip)
        
        # Composite video with subtitles
        print("[INFO] Compositing video with subtitles...")
        final_video = CompositeVideoClip([video_clip] + subtitle_clips)
        
        # Generate output filename
        output_filename = f"captioned_{uuid.uuid4().hex[:8]}.mp4"
        output_path = f"/data/{output_filename}"
        
        # Write final video
        print("[INFO] Writing final video...")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='/data/temp-audio.m4a',
            remove_temp=True
        )
        
        # Clean up
        video_clip.close()
        final_video.close()
        os.unlink(temp_video_path)
        
        print(f"[INFO] Video saved to {output_path}")
        
        # Return the video file
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

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    memory=4096,
    cpu=2.0,
    secrets=[modal.Secret.from_name("openai-api-key")]
)
@modal.fastapi_endpoint(method="POST")
async def generate_live_subtitles(
    video: UploadFile = File(...),
    language: str = Form("en"),
    font: str = Form("arial")
):
    """Generate live karaoke-style subtitles with word-level timing. Font: 'arial', 'georgia', 'montserrat', 'verdana', 'comic_sans' (default: 'arial')"""
    import os
    import tempfile
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import httpx
    import asyncio
    import json
    import uuid
    
    # Set up environment
    os.environ["MAGICK_HOME"] = "/usr"
    os.environ["PATH"] = f"{os.environ['MAGICK_HOME']}/bin:" + os.environ.get("PATH", "")
    
    FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
    font_map = {
        "arial": os.path.join(FONT_DIR, "Arial-Bold.ttf"),
        "georgia": os.path.join(FONT_DIR, "Georgia-Bold.ttf"),
        "montserrat": os.path.join(FONT_DIR, "Montserrat-Bold.ttf"),
        "verdana": os.path.join(FONT_DIR, "Verdana-Bold.ttf"),
        "comic_sans": os.path.join(FONT_DIR, "ComicSansMS-Bold.ttf"),
        "times_new_roman": os.path.join(FONT_DIR, "TimesNewRoman-Bold.ttf"),
        "courier_new": os.path.join(FONT_DIR, "CourierNew-Bold.ttf"),
        "trebuchet_ms": os.path.join(FONT_DIR, "TrebuchetMS-Bold.ttf"),
        "tahoma": os.path.join(FONT_DIR, "Tahoma-Bold.ttf"),
    }
    
    try:
        print("[INFO] Starting live subtitle generation...")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, dir="/data", suffix=".mp4") as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        
        # Load the video
        video_clip = VideoFileClip(temp_video_path)
        
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
        
        file_size = os.path.getsize(temp_video_path)
        timeout_seconds = min(30 + (file_size / (1024 * 1024)), 300)
        
        with open(temp_video_path, "rb") as audio_file:
            files = {"file": (os.path.basename(temp_video_path), audio_file, "audio/mp3")}
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
        
        def create_karaoke_image(text, highlighted_word, video_width):
            """Create karaoke-style image with highlighted word and selected font"""
            font_path = font_map.get(font.lower(), font_map["arial"])
            print(f"[FONT DEBUG] Requested font: {font}, Path: {font_path}")
            try:
                font_obj = ImageFont.truetype(font_path, 48)
                print(f"[FONT DEBUG] Successfully loaded font: {font_path}")
            except Exception as e:
                print(f"[ERROR] Could not load font {font_path}: {e}")
                font_obj = ImageFont.load_default()
                print(f"[FONT DEBUG] Fallback to default font.")
            img = Image.new('RGBA', (video_width, 150), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Split text into words
            words = text.split()
            current_x = 50
            line_height = 60
            for i, word in enumerate(words):
                if word == highlighted_word:
                    color = "#00A5FF"
                else:
                    color = "black"
                draw.text((current_x, 45), word, fill=color, font=font_obj)
                bbox = draw.textbbox((current_x, 45), word, font=font_obj)
                current_x = bbox[2] + 20
                if current_x > video_width - 100:
                    current_x = 50
                    line_height += 60
            return np.array(img)
        
        # Group words into sentences and create clips
        current_sentence = []
        sentence_start = None
        
        for word_data in words:
            word = word_data["word"]
            start = word_data["start"]
            end = word_data["end"]
            
            if sentence_start is None:
                sentence_start = start
            
            current_sentence.append(word)
            
            # Create clip for this word
            sentence_text = " ".join(current_sentence)
            text_clip = TextClip(
                lambda t: create_karaoke_image(sentence_text, word, video_clip.w),
                duration=end - start,
                size=(video_clip.w, 150)
            )
            text_clip = text_clip.set_position(("center", "bottom")).set_start(start)
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
            temp_audiofile='/data/temp-audio.m4a',
            remove_temp=True
        )
        
        # Clean up
        video_clip.close()
        final_video.close()
        os.unlink(temp_video_path)
        
        # Return the video file
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