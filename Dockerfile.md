# Dockerfile

A minimal image that:
- Uses python:3.11-slim (or similar)
- Installs ffmpeg
- Installs Python dependencies from requirements.txt
- Copies project files
- Exposes FastAPI example by default (or keep it as a sample command)

# docker-compose.yml

Simple service runner for local dev:

- Builds from Dockerfile
- Maps port 8000
- Passes through OPENAI_API_KEY environment variable
