# video_analyzer/__init__.py

"""
Video AI Analyzer

Public API re-exports for convenient imports:
- analyze_video_file(file_obj, filename=None) -> str
- analyze_video_bytes(video_bytes, filename="input.mp4") -> str
- start_log_capture(), get_captured_logs(), end_log_capture()
"""

from .video_analyzer import (
    analyze_video_file,
    analyze_video_bytes,
    start_log_capture,
    get_captured_logs,
    end_log_capture,
)

__all__ = [
    "analyze_video_file",
    "analyze_video_bytes",
    "start_log_capture",
    "get_captured_logs",
    "end_log_capture",
]

__version__ = "0.1.0"
