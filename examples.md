- fastapi_app.py

      Minimal FastAPI app with a single POST /analyze endpoint.    

      Accepts UploadFile; passes upload.file and upload.filename into analyze_video_file.   

      Optionally captures logs and returns them to the client.

- flask_app.py

      Minimal Flask app with POST /analyze endpoint.
      Accepts request.files["file"]; passes .stream and filename into analyze_video_file.

- cli.py

      Command-line tool for local testing.
      Reads a local path just for demo, then passes bytes into analyze_video_bytes.
