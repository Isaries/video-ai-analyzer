# video-ai-analyzer
Process user-uploaded videos on the server without persistent storage: extract representative frames for vision analysis, transcribe the audio, and produce a unified summary. Designed for web backends with file upload, calling straight from an in-memory file object or bytes.

The library uses a short-lived temporary file only during processing and deletes it immediately after.

- Input: file-like object (e.g., from a web framework) or raw bytes
- Output: a textual summary (vision only, or vision + audio if present)
- OpenAI API key: read from environment variable OPENAI_API_KEY
- Temporary storage: uses OS temp directory for the video during processing; not persisted

# Features
- Accepts uploads directly as file-like objects or bytes (no need to save to your storage)
- Frame extraction in memory (JPEG bytes), combined with OpenAI vision
- Audio detection and in-memory transcription (WAV bytes -> OpenAI transcription)
- Merges vision and audio into a final summary
- Minimal API surface for backend developers
- Logs you can optionally capture and return to the client for progress feedback

# How It Works
- Your backend passes the uploaded video as a file object or bytes.
- The library writes it to a temporary file (OS temp dir) to support tools like ffmpeg/ffprobe/OpenCV/MoviePy that depend on random-access paths.
- Frames are extracted to memory (no image files).
- Audio, if present, is extracted to WAV bytes in memory and sent to transcription.
- After processing, the temporary video file is deleted.
- No persistent storage to your server or cloud is performed by this library.

If your environment forbids any disk writes at all (including temporary), see “Zero-Disk Environments”.

# High-level goals
- Accept uploaded videos as file-like objects or bytes in backend servers.
- Use a short-lived temp file during processing (deleted afterward).
- Extract frames (in memory), detect/extract audio (in memory), call OpenAI for vision and transcription, and output a final summary.
- Provide ready-to-run examples (FastAPI, Flask, CLI) and optional Docker setup.

# Tech stack
- Python 3.10+
- ffmpeg/ffprobe (system tools)
- OpenAI Python SDK
- MoviePy, Pillow, OpenCV (headless)
- FastAPI/Uvicorn and Flask (examples)
- Optional Docker

# Public API surface (what backend developers use)

- analyze_video_file(file_obj, filename=None) -> str
- Pass a binary file-like object and optionally the original filename (for extension inference).

- analyze_video_bytes(video_bytes, filename="input.mp4") -> str
- Pass raw bytes and a filename hint for extension inference.

- start_log_capture(), get_captured_logs(), end_log_capture()
- Optional logging helpers to stream progress back to clients.

# Key behaviors and constraints

- Uses OS temp directory to store a short-lived copy of the uploaded video; required by ffmpeg/ffprobe/OpenCV/MoviePy.
- Extracted frames and audio are handled in memory; no image files are written.
- Temporary video is deleted after processing; nothing is persisted by this library.
- Requires ffmpeg/ffprobe in PATH for best results.
- Reads OpenAI API key from OPENAI_API_KEY.

# Build and run

- Local:
> python -m venv .venv
> source .venv/bin/activate (Windows: .venv\Scripts\activate)
> pip install -r requirements.txt
> export/setx OPENAI_API_KEY

- CLI:
> python examples/cli.py demo.mp4 --name demo.mp4

- FastAPI:
> uvicorn examples.fastapi_app:app --host 0.0.0.0 --port 8000
> curl -X POST "http://localhost:8000/analyze" -F "upload=@demo.mp4"

- Flask:
> FLASK_APP=examples.flask_app flask run --host=0.0.0.0 --port=8000
> curl -X POST "http://localhost:8000/analyze" -F "file=@demo.mp4"

# Docker usage

- Build: docker build -t video-ai-analyzer .
- Run: docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8000:8000 video-ai-analyzer
- Compose:
> docker-compose up --build

# Testing suggestions

- Unit tests for:
> Extension inference and temp file creation/deletion
> Frame timestamp generation logic
> Logging capture utilities
- Integration tests (mark as slow/optional):
> Require ffmpeg and minimal sample video
> Mock OpenAI calls or use a test account with rate-limited runs

# Security and privacy notes

- The library does not persist videos; it creates a temporary file during processing and deletes it afterward.
- It sends frame images and audio snippets to OpenAI. You are responsible for informing users and complying with your policies and OpenAI’s terms.
- Validate upload types and sizes at the gateway or app layer.

# Roadmap

- Zero-disk (pipe-only) mode for environments with no write privileges
- Structured output (JSON) with timestamps and entities
- Advanced frame sampling strategies and GPU-accelerated pipelines
- More robust error handling and retry policies
