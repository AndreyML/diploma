"""
Database models for the AI Content Generation Service
Based on the schema defined in the diploma work
"""
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey, JSON, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
Base = declarative_base()
class User(Base):
    """User model - stores user information"""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False, index=True)
    created_at = Column(TIMESTAMP, default=func.now())
    # Relationships
    image_generations = relationship("ImageGeneration", back_populates="user")
    model_trainings = relationship("ModelTraining", back_populates="user")
    lora_adapters = relationship("LoRAAdapter", back_populates="user")
class ImageGeneration(Base):
    """Image generation history model"""
    __tablename__ = "image_generations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    prompt = Column(Text, nullable=False)
    lora_path = Column(Text, nullable=True)
    num_images = Column(Integer, default=1)
    images_s3_urls = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, default=func.now())
    # Relationships
    user = relationship("User", back_populates="image_generations")
class ModelTraining(Base):
    """Model training history model"""
    __tablename__ = "model_trainings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    train_data = Column(JSON, nullable=False)  # List of {image: base64, description: str}
    lora_params_url = Column(Text, nullable=True)
    status = Column(String(50), default="started")  # started, completed, error
    created_at = Column(TIMESTAMP, default=func.now())
    # Relationships
    user = relationship("User", back_populates="model_trainings")
class LoRAAdapter(Base):
    """LoRA adapter model"""
    __tablename__ = "lora_adapters"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    s3_url = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, default=func.now())
    # Relationships
    user = relationship("User", back_populates="lora_adapters")
