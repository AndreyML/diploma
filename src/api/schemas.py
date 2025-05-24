"""
Pydantic schemas for REST API
Based on the API specification from the diploma work
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
# Image Generation Schemas
class ImageGenerationRequest(BaseModel):
    """Request schema for image generation endpoint"""
    prompt: str = Field(..., description="Text prompt for image generation", min_length=1, max_length=1000)
    lora_path: Optional[str] = Field(None, description="Path or URL to LoRA adapter")
    num_images: int = Field(1, description="Number of images to generate", ge=1, le=4)
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()
class ImageGenerationResponse(BaseModel):
    """Response schema for image generation endpoint"""
    images: List[str] = Field(..., description="List of base64 encoded images")
# VLM Schemas
class ImageDescriptionRequest(BaseModel):
    """Request schema for image description endpoint"""
    images: List[str] = Field(..., description="List of base64 encoded images", min_items=1, max_items=10)
    @validator('images')
    def validate_images(cls, v):
        if not v:
            raise ValueError('At least one image is required')
        return v
class ImageDescriptionResponse(BaseModel):
    """Response schema for image description endpoint"""
    descriptions: List[str] = Field(..., description="List of generated descriptions")
# LLM Schemas
class PromptImprovementRequest(BaseModel):
    """Request schema for prompt improvement endpoint"""
    prompt: str = Field(..., description="Original text prompt", min_length=1, max_length=500)
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()
class PromptImprovementResponse(BaseModel):
    """Response schema for prompt improvement endpoint"""
    improved_prompt: str = Field(..., description="Improved version of the prompt")
# Training Schemas
class TrainingDataItem(BaseModel):
    """Single training data item"""
    image: str = Field(..., description="Base64 encoded image")
    description: str = Field(..., description="Text description of the image", min_length=1, max_length=500)
class TrainingRequest(BaseModel):
    """Request schema for model training endpoint"""
    train_data: List[TrainingDataItem] = Field(..., description="List of training data items", min_items=1, max_items=100)
    @validator('train_data')
    def validate_train_data(cls, v):
        if len(v) < 1:
            raise ValueError('At least one training sample is required')
        return v
class TrainingResponse(BaseModel):
    """Response schema for model training endpoint"""
    lora_params_url: str = Field(..., description="S3 URL of trained LoRA weights")
    training_id: Optional[str] = Field(None, description="Training job ID for async training")
# User Schemas
class UserCreate(BaseModel):
    """Schema for creating a new user"""
    telegram_id: int = Field(..., description="Telegram user ID")
class UserResponse(BaseModel):
    """Response schema for user information"""
    id: int
    telegram_id: int
    created_at: datetime
    class Config:
        from_attributes = True
# History Schemas
class ImageGenerationHistory(BaseModel):
    """Schema for image generation history"""
    id: int
    prompt: str
    lora_path: Optional[str]
    num_images: int
    images_s3_urls: Optional[List[str]]
    created_at: datetime
    class Config:
        from_attributes = True
class TrainingHistory(BaseModel):
    """Schema for training history"""
    id: int
    lora_params_url: Optional[str]
    status: str
    created_at: datetime
    class Config:
        from_attributes = True
# Error Schemas
class ErrorResponse(BaseModel):
    """Standard error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
class ValidationErrorResponse(BaseModel):
    """Validation error response schema"""
    error: str = "Validation Error"
    details: List[Dict[str, Any]] = Field(..., description="List of validation errors")
# Status Schemas
class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
class ServiceStatusResponse(BaseModel):
    """Service status response schema"""
    api_status: str = Field(..., description="API service status")
    database_status: str = Field(..., description="Database connection status")
    s3_status: str = Field(..., description="S3 storage status")
    models_status: Dict[str, str] = Field(..., description="AI models status")
# Configuration Schemas
class ModelConfigResponse(BaseModel):
    """Model configuration response schema"""
    flux_model: str
    vlm_model: str
    llm_model: str
    lora_rank: int
    batch_size: int
    max_images_per_request: int
