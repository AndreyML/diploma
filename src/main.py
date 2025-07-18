"""
Main FastAPI application
AI Content Generation Service for Marketplaces
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
from src.config.settings import settings
from src.database import init_db
from src.api.endpoints import router
from src.api.schemas import ErrorResponse
# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI Content Generation Service...")
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    # Initialize AI models (commented out for initial setup)
    # In production, you might want to preload models here
    # try:
    #     from src.models import vlm_module, llm_module, flux_module
    #     vlm_module.load_model()
    #     llm_module.load_model()
    #     flux_module.load_pipeline()
    #     logger.info("AI models loaded successfully")
    # except Exception as e:
    #     logger.warning(f"AI models loading failed: {e}")
    logger.info("Service startup completed")
    yield
    logger.info("Shutting down AI Content Generation Service...")
# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    AI Content Generation Service for Marketplaces - Diploma Project
    This service provides AI-powered content generation capabilities including:
    - Image generation using FLUX.1-dev with LoRA adaptation
    - Image description using Visual-Language Models (VLM)
    - Prompt enhancement using Large Language Models (LLM)
    - Custom model training with LoRA

    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include API router
app.include_router(router, prefix="/api/v1", tags=["AI Content Generation"])
# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            code="INTERNAL_ERROR"
        ).dict()
    )
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "AI Content Generation Service",
        "version": settings.API_VERSION,
        "description": "Diploma project implementation",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "/api/v1/status"
    }
# Additional utility endpoints
@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong"}
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
