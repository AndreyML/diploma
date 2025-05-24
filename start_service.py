#!/usr/bin/env python3
"""
Service startup script for AI Content Generation Service
Handles environment setup and service initialization
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def setup_environment():
    """Set up environment variables for development"""
    env_vars = {
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
        "API_RELOAD": "true",
        "DATABASE_URL": "sqlite:///./ai_content.db",
        "LOG_LEVEL": "INFO",
        "LOG_FILE": "app.log",
        # Mock S3 settings for testing
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "S3_BUCKET_NAME": "test-bucket",
        "S3_ENDPOINT_URL": "http://localhost:9000",
        # Mock Redis for testing
        "REDIS_URL": "redis://localhost:6379/0",
        # Model paths (will use smaller models for testing)
        "FLUX_MODEL_PATH": "black-forest-labs/FLUX.1-schnell",  # Faster model for testing
        "VLM_MODEL_PATH": "Qwen/Qwen2-VL-2B-Instruct",  # Smaller model for testing
        "LLM_MODEL_PATH": "Qwen/Qwen2.5-7B-Instruct",  # Smaller model for testing
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"Set {key}={value}")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import transformers
        import diffusers
        import torch
        import PIL
        print("✅ All required dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -e .")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting AI Content Generation Service...")
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        import uvicorn
        from src.main import app
        
        uvicorn.run(
            app,
            host=os.environ.get("API_HOST", "0.0.0.0"),
            port=int(os.environ.get("API_PORT", "8000")),
            reload=os.environ.get("API_RELOAD", "true").lower() == "true",
            log_level=os.environ.get("LOG_LEVEL", "info").lower()
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("AI Content Generation Service - Startup Script")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the service
    start_api_server()

if __name__ == "__main__":
    main() 