"""
Configuration settings for the AI Content Generation Service
"""
from pydantic_settings import BaseSettings
from typing import Optional, Tuple


class Settings(BaseSettings):
    """Application settings"""

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_TITLE: str = "AI Content Generation Service"
    API_VERSION: str = "1.0.0"

    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/ai_content_db"
    DATABASE_ECHO: bool = False

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "ai-content-storage"
    S3_ENDPOINT_URL: Optional[str] = None

    REDIS_URL: str = "redis://localhost:6379/0"

    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_WEBHOOK_URL: Optional[str] = None

    FLUX_MODEL_PATH: str = "black-forest-labs/FLUX.1-dev"
    VLM_MODEL_PATH: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    LLM_MODEL_PATH: str = "Qwen/Qwen2.5-32B-Instruct"

    LORA_RANK: int = 16
    LORA_ALPHA: int = 16
    LEARNING_RATE: float = 1e-4
    BATCH_SIZE: int = 32
    TRAINING_STEPS: int = 1000

    FLUX_LORA_RANK: int = 16
    FLUX_LORA_ALPHA: int = 16
    FLUX_LEARNING_RATE: float = 1e-4
    FLUX_BATCH_SIZE: int = 1
    FLUX_EPOCHS: int = 1
    FLUX_MAX_SEQUENCE_LENGTH: int = 512
    FLUX_CLIP_MAX_LENGTH: int = 77

    QUANTIZE_TEXT_ENCODERS: bool = True
    LOW_VRAM_MODE: bool = False
    GRADIENT_CHECKPOINTING: bool = True
    MIXED_PRECISION: bool = True

    GRADIENT_CLIP_NORM: float = 1.0
    WEIGHT_DECAY: float = 0.01
    WARMUP_STEPS: int = 100
    SAVE_CHECKPOINT_EVERY: int = 500
    LOG_EVERY: int = 100

    FLOW_MATCHING_TIMESTEPS: int = 1000
    NOISE_OFFSET: float = 0.0

    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
    MAX_IMAGES_PER_REQUEST: int = 4

    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
