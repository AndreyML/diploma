"""
REST API endpoints implementation
Based on the API specification from the diploma work
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List
import logging
from datetime import datetime
from src.database import get_db, User, ImageGeneration, ModelTraining
from src.models import vlm_module, llm_module, flux_module, training_module
from src.config.settings import settings
from src.api.schemas import (
    ImageGenerationRequest, ImageGenerationResponse,
    ImageDescriptionRequest, ImageDescriptionResponse,
    PromptImprovementRequest, PromptImprovementResponse,
    TrainingRequest, TrainingResponse,
    HealthCheckResponse, ServiceStatusResponse, ModelConfigResponse,
    ErrorResponse
)
logger = logging.getLogger(__name__)
# Create router
router = APIRouter()
# Health check endpoints
@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Basic health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.API_VERSION
    )
@router.get("/status", response_model=ServiceStatusResponse)
async def service_status(db: Session = Depends(get_db)):
    """Detailed service status check"""
    try:
        # Check database
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    # Check S3 (simplified)
    s3_status = "healthy"  # In practice, test S3 connectivity
    # Check models status
    models_status = {
        "vlm": "loaded" if vlm_module.model is not None else "not_loaded",
        "llm": "loaded" if llm_module.model is not None else "not_loaded",
        "flux": "loaded" if flux_module.pipeline is not None else "not_loaded"
    }
    return ServiceStatusResponse(
        api_status="healthy",
        database_status=db_status,
        s3_status=s3_status,
        models_status=models_status
    )
@router.get("/config", response_model=ModelConfigResponse)
async def get_config():
    """Get current model configuration"""
    return ModelConfigResponse(
        flux_model=settings.FLUX_MODEL_PATH,
        vlm_model=settings.VLM_MODEL_PATH,
        llm_model=settings.LLM_MODEL_PATH,
        lora_rank=settings.LORA_RANK,
        batch_size=settings.BATCH_SIZE,
        max_images_per_request=settings.MAX_IMAGES_PER_REQUEST
    )
# Image generation endpoint
@router.post("/generate_image", response_model=ImageGenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate images from text prompt
    Main endpoint implementing the image generation pipeline
    """
    try:
        logger.info(f"Generating {request.num_images} images for prompt: {request.prompt[:50]}...")
        
        if request.num_images > settings.MAX_IMAGES_PER_REQUEST:
            raise HTTPException(
                status_code=400, 
                detail=f"Number of images cannot exceed {settings.MAX_IMAGES_PER_REQUEST}"
            )
        
        images = flux_module.generate_images(
            prompt=request.prompt,
            lora_path=request.lora_path,
            num_images=request.num_images
        )

        dummy_user_id = 1
        try:
            generation_record = ImageGeneration(
                user_id=dummy_user_id,
                prompt=request.prompt,
                lora_path=request.lora_path,
                num_images=request.num_images,
                images_s3_urls=[]
            )
            db.add(generation_record)
            db.commit()
            logger.info("Generation history saved to database")
        except SQLAlchemyError as e:
            logger.warning(f"Failed to save generation history: {e}")
            db.rollback()

        logger.info(f"Successfully generated {len(images)} images")
        return ImageGenerationResponse(images=images)
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
# Image description endpoint (VLM)
@router.post("/describe_image", response_model=ImageDescriptionResponse)
async def describe_image(request: ImageDescriptionRequest):
    """
    Generate descriptions for images using VLM
    Implementation of the VLM module from the diploma
    """
    try:
        logger.info(f"Describing {len(request.images)} images")
        
        if len(request.images) > 10:
            raise HTTPException(
                status_code=400,
                detail="Cannot process more than 10 images at once"
            )
        
        descriptions = vlm_module.describe_images(request.images)
        logger.info(f"Successfully generated {len(descriptions)} descriptions")
        return ImageDescriptionResponse(descriptions=descriptions)
    except Exception as e:
        logger.error(f"Error describing images: {e}")
        raise HTTPException(status_code=500, detail=f"Image description failed: {str(e)}")
# Prompt improvement endpoint (LLM)
@router.post("/improve_prompt", response_model=PromptImprovementResponse)
async def improve_prompt(request: PromptImprovementRequest):
    """
    Improve user prompt using LLM
    Implementation of the LLM module from the diploma
    """
    try:
        logger.info(f"Improving prompt: {request.prompt[:50]}...")
        improved_prompt = llm_module.improve_prompt(request.prompt)
        logger.info(f"Successfully improved prompt")
        return PromptImprovementResponse(improved_prompt=improved_prompt)
    except Exception as e:
        logger.error(f"Error improving prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Prompt improvement failed: {str(e)}")
# Model training endpoint
@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Train LoRA adapter on provided data
    Implementation of the training module from the diploma
    """
    try:
        logger.info(f"Starting LoRA training with {len(request.train_data)} samples")
        
        if len(request.train_data) > 100:
            raise HTTPException(
                status_code=400,
                detail="Cannot train with more than 100 samples at once"
            )
        
        train_data = [
            {"image": item.image, "description": item.description}
            for item in request.train_data
        ]

        dummy_user_id = 1
        lora_params_url = training_module.train_lora(train_data, dummy_user_id)

        try:
            training_record = ModelTraining(
                user_id=dummy_user_id,
                train_data=train_data,
                lora_params_url=lora_params_url,
                status="completed"
            )
            db.add(training_record)
            db.commit()
            logger.info("Training history saved to database")
        except SQLAlchemyError as e:
            logger.warning(f"Failed to save training history: {e}")
            db.rollback()

        logger.info(f"LoRA training completed successfully")
        return TrainingResponse(lora_params_url=lora_params_url)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
# Combined generation and training endpoint
@router.post("/train_model", response_model=TrainingResponse)
async def train_model_alias(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Alias for /train endpoint (as specified in the diploma)
    """
    return await train_model(request, background_tasks, db)
# Advanced generation endpoint with automatic prompt improvement
@router.post("/generate_enhanced", response_model=ImageGenerationResponse)
async def generate_enhanced_image(
    request: ImageGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate images with automatic prompt enhancement
    Combines LLM prompt improvement + FLUX generation
    """
    try:
        logger.info(f"Enhanced generation for prompt: {request.prompt[:50]}...")
        
        if request.num_images > settings.MAX_IMAGES_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Number of images cannot exceed {settings.MAX_IMAGES_PER_REQUEST}"
            )
        
        improved_prompt = llm_module.improve_prompt(request.prompt)
        logger.info(f"Prompt improved: {improved_prompt[:50]}...")
        
        images = flux_module.generate_images(
            prompt=improved_prompt,
            lora_path=request.lora_path,
            num_images=request.num_images
        )

        dummy_user_id = 1
        try:
            generation_record = ImageGeneration(
                user_id=dummy_user_id,
                prompt=f"Original: {request.prompt} | Improved: {improved_prompt}",
                lora_path=request.lora_path,
                num_images=request.num_images,
                images_s3_urls=[]
            )
            db.add(generation_record)
            db.commit()
        except SQLAlchemyError as e:
            logger.warning(f"Failed to save generation history: {e}")
            db.rollback()

        logger.info(f"Enhanced generation completed successfully")
        return ImageGenerationResponse(images=images)
    except Exception as e:
        logger.error(f"Error in enhanced generation: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced generation failed: {str(e)}")
# Utility endpoints for development/testing
@router.post("/load_models")
async def load_models():
    """Load all AI models (utility endpoint for development)"""
    try:
        logger.info("Loading all AI models...")
        # Load models
        vlm_module.load_model()
        llm_module.load_model()
        flux_module.load_pipeline()
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
@router.get("/models/status")
async def get_models_status():
    """Get current status of all models"""
    return {
        "vlm_loaded": vlm_module.model is not None,
        "llm_loaded": llm_module.model is not None,
        "flux_loaded": flux_module.pipeline is not None
    }
