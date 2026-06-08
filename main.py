#!/usr/bin/env python3
"""
Chatterbox TTS API Entry Point

This is the main entry point for the application.
It imports the FastAPI app from the organized app package.
"""

# chatterbox/utils/splitter.py calls nltk.download() at import time, which
# tries to fetch the NLTK index from the internet before checking local data.
# This hangs indefinitely in network-restricted environments. punkt_tab is
# pre-installed during the Docker build, so the network check is unnecessary.
import nltk
nltk.download = lambda *args, **kwargs: True

import uvicorn
from app.main import app
from app.config import Config


def main():
    """Main entry point"""
    try:
        Config.validate()
        print(f"Starting Chatterbox TTS API server...")
        print(f"Server will run on http://{Config.HOST}:{Config.PORT}")
        print(f"API documentation available at http://{Config.HOST}:{Config.PORT}/docs")
        
        uvicorn.run(
            "app.main:app",
            host=Config.HOST,
            port=Config.PORT,
            reload=False,
            access_log=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        exit(1)


if __name__ == "__main__":
    main() 