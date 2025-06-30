# Force Modal rebuild - minimal test for openai==1.2.4 and pydantic==1.10.13
import os
from moviepy.config import change_settings
IMAGEMAGICK_BINARY = "/usr/bin/convert"
change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})
import modal
from pathlib import Path
from fastapi import UploadFile, File

# Create a Modal app
app = modal.App("auto-caption-generator")

# Define the image with all necessary dependencies
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "imagemagick",
    "ffmpeg", 
    "fonts-dejavu",
    "fonts-liberation",
    "fonts-noto"
).run_commands(
    # Completely disable ImageMagick security policy
    "cp /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.backup",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"PDF\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"PDF\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"LABEL\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"LABEL\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"TEXT\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"TEXT\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"PS\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"PS\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"EPS\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"EPS\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"XPS\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"XPS\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"MVG\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"MVG\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"SVG\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"SVG\"\/> -->/g' /etc/ImageMagick-6/policy.xml",
    "sed -i 's/<policy domain=\"coder\" rights=\"none\" pattern=\"XML\"\/>/<!-- <policy domain=\"coder\" rights=\"none\" pattern=\"XML\"\/> -->/g' /etc/ImageMagick-6/policy.xml"
).pip_install(
    "openai==1.2.4",
    "pydantic==1.10.13",
    "fastapi[standard]==0.104.1",
    "python-multipart==0.0.6",
    "moviepy==1.0.3",
    "httpx==0.23.3"
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
    language: str = "en",
    style: str = "classic"
):
    """Generate subtitles for uploaded video"""
    import os
    import tempfile
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, VideoClip
    from moviepy.video.tools.subtitles import SubtitlesClip
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    from openai import OpenAI
    import httpx
    import asyncio
    import json
    import uuid
    import textwrap
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    
    # Set up environment
    os.environ["MAGICK_HOME"] = "/usr"
    os.environ["PATH"] = f"{os.environ['MAGICK_HOME']}/bin:" + os.environ.get("PATH", "")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Subtitle presets
    SUBTITLE_PRESETS = {
        "youtube": {
            "font": "DejaVuSans-Bold",
            "fontsize": 48,
            "color": "white",
            "highlight_color": "#00A5FF",
            "bg_color": "rgba(0,0,0,0.7)",
            "bg_opacity": 0.7,
            "position": ("center", "bottom"),
            "method": "label",
            "align": "center",
            "spacing": 5
        },
        "classic": {
            "font": "DejaVuSans-Bold",
            "fontsize": 48,
            "color": "white",
            "highlight_color": "#00A5FF",
            "bg_color": None,
            "white_bg": True,
            "bg_padding": 20,
            "position": ("center", "bottom"),
            "method": "label",
            "align": "center",
            "stroke_color": "black",
            "stroke_width": 2,
            "spacing": 5
        },
        "centered": {
            "font": "DejaVuSans-Bold",
            "fontsize": 48,
            "color": "white",
            "highlight_color": "#00A5FF",
            "bg_color": "rgba(0,0,0,0.7)",
            "bg_opacity": 0.7,
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
            temp_video.write(video.file.read())
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
            """Create a text clip with proper styling and background, matching reference and screenshots"""
            def make_frame(t):
                # Set max text width to 80% of video width
                max_text_width = int(video_width * 0.8)
                font_size = style_settings["fontsize"]
                font_name = style_settings["font"]
                # Font fallback logic
                font_paths = [
                    f"/usr/share/fonts/truetype/dejavu/{font_name}.ttf",
                    f"/usr/share/fonts/truetype/dejavu/{font_name.lower()}.ttf",
                    f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ]
                font = None
                for fp in font_paths:
                    try:
                        font = ImageFont.truetype(fp, font_size)
                        break
                    except Exception:
                        continue
                if font is None:
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
                # Background logic
                is_white_bg = style_settings.get("white_bg", False)
                if is_white_bg:
                    bg_padding = style_settings.get("bg_padding", 20)
                    rect_x0 = (img_w - max_width) // 2 - bg_padding
                    rect_y0 = (img_h - total_height) // 2 - bg_padding
                    rect_x1 = (img_w + max_width) // 2 + bg_padding
                    rect_y1 = (img_h + total_height) // 2 + bg_padding
                    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(255, 255, 255, 255))
                elif style_settings.get("bg_color"):
                    # Semi-transparent bg
                    bg_margin_x = 30
                    bg_margin_y = 20
                    rect_x0 = (img_w - max_width) // 2 - bg_margin_x
                    rect_y0 = (img_h - total_height) // 2 - bg_margin_y
                    rect_x1 = (img_w + max_width) // 2 + bg_margin_x
                    rect_y1 = (img_h + total_height) // 2 + bg_margin_y
                    # Parse rgba string
                    bg_color = style_settings["bg_color"]
                    if bg_color.startswith("rgba"):
                        vals = bg_color[5:-1].split(",")
                        r, g, b, a = [float(x.strip()) for x in vals]
                        fill = (int(r), int(g), int(b), int(a * 255) if a <= 1 else int(a))
                    else:
                        fill = (0, 0, 0, 180)
                    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=fill)
                # Draw each line with stroke and highlight
                y = (img_h - total_height) // 2
                for line in lines:
                    words_in_line = line.split()
                    word_sizes = []
                    for word in words_in_line:
                        bbox = draw.textbbox((0, 0), word + " ", font=font)
                        word_sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    total_line_width = sum(w for w, h in word_sizes) + style_settings["spacing"] * (len(words_in_line) - 1)
                    x = (img_w - total_line_width) // 2
                    current_x = x
                    for i, (word, (w, h)) in enumerate(zip(words_in_line, word_sizes)):
                        # Stroke for classic
                        if style_settings.get("stroke_width", 0) > 0 and style_settings.get("stroke_color"):
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
                    y += h + 10
                img_array = np.array(img)
                rgb = img_array[..., :3]
                alpha = img_array[..., 3]
                mask = alpha / 255.0
                return rgb * mask[..., np.newaxis]
            from moviepy.video.VideoClip import VideoClip
            return VideoClip(lambda t: make_frame(t), duration=1)
        
        # Process segments and create subtitle clips
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
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
            temp_audiofile='temp-audio.m4a',
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
    language: str = "en"
):
    """Generate live karaoke-style subtitles with word-level timing"""
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
    
    try:
        print("[INFO] Starting live subtitle generation...")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, dir="/data", suffix=".mp4") as temp_video:
            temp_video.write(video.file.read())
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
        
        def create_karaoke_clip(chunk_text, highlight_word, video_width, duration):
            """Create a PIL karaoke subtitle image (all-caps, bold, white bg, black text, blue highlight, centered, with padding, fixed 48px font, and up to 2 lines)"""
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font_size = 48
            padding = 50
            max_bar_width = int(video_width * 0.95)
            chunk_text = chunk_text.strip().upper()
            highlight_word = highlight_word.strip().upper()
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
            # Word wrapping: split into lines so each line fits max_bar_width - 2*padding
            words = chunk_text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = (current_line + " " + word).strip()
                bbox = font.getbbox(test_line)
                w = bbox[2] - bbox[0]
                if w > (max_bar_width - 2 * padding) and current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            # Only allow up to 2 lines
            if len(lines) > 2:
                lines = [" ".join(words[:len(words)//2]), " ".join(words[len(words)//2:])]
            # Calculate bar size
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            text_block_height = sum(line_heights) + (len(lines) - 1) * 10
            bar_height = text_block_height + 2 * padding
            max_line_width = max([font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines])
            img_width = max(max_bar_width, max_line_width + 2 * padding)
            img = Image.new('RGBA', (img_width, bar_height), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Draw lines, highlight current word
            y = padding
            for idx, line in enumerate(lines):
                line_words = line.split()
                line_bbox = font.getbbox(line)
                line_width = line_bbox[2] - line_bbox[0]
                x = (img_width - line_width) // 2
                current_x = x
                for word in line_words:
                    clean_word = word.strip('.,!?;:')
                    clean_highlighted = highlight_word.strip('.,!?;:')
                    word_bbox = font.getbbox(word + " ")
                    word_width = word_bbox[2] - word_bbox[0]
                    word_color = (0, 165, 255) if (word == highlight_word or clean_word == clean_highlighted) else (0, 0, 0)
                    # Draw bold text (draw twice with 1px offset)
                    draw.text((current_x, y), word, font=font, fill=word_color)
                    draw.text((current_x+1, y), word, font=font, fill=word_color)
                    current_x += word_width
                y += line_heights[idx] + 10
            # Center the bar horizontally in the video
            def make_frame(t):
                rgb_img = Image.new('RGB', (video_width, bar_height), (255, 255, 255))
                x_offset = (video_width - img_width) // 2
                rgb_img.paste(img, (x_offset, 0), mask=img.split()[-1])
                return np.array(rgb_img)
            from moviepy.video.VideoClip import VideoClip
            return VideoClip(make_frame, duration=duration)

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
                text_clip = create_karaoke_clip(chunk_text, highlight_word, video_clip.w, duration)
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