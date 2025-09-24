# video-ai-analyzer
Process user-uploaded videos on the server without persistent storage: extract representative frames for vision analysis, transcribe the audio, and produce a unified summary. Designed for web backends with file upload, calling straight from an in-memory file object or bytes.

The library uses a short-lived temporary file only during processing and deletes it immediately after.

- Input: file-like object (e.g., from a web framework) or raw bytes
- Output: a textual summary (vision only, or vision + audio if present)
- OpenAI API key: read from environment variable OPENAI_API_KEY
- Temporary storage: uses OS temp directory for the video during processing; not persisted
