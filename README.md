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
