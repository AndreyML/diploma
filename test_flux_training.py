#!/usr/bin/env python3
"""
Test script for FLUX LoRA training implementation
"""

from src.models.flux_training_utils import (
    validate_flux_training_config,
    estimate_flux_memory_usage,
    get_flux_target_modules
)
from src.models.training_module import flux_training_module
import sys
import base64
import logging
from PIL import Image
import io

sys.path.append('src')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_image(size=(1024, 1024), color=(255, 0, 0)):
    """Create a test image and return as base64"""
    image = Image.new('RGB', size, color)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')


def test_flux_training_utils():
    """Test FLUX training utilities"""
    logger.info("Testing FLUX training utilities...")

    try:
        validate_flux_training_config(
            batch_size=1,
            learning_rate=1e-4,
            rank=16,
            alpha=16
        )
        logger.info("✓ Configuration validation passed")
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False

    try:
        memory_info = estimate_flux_memory_usage(
            batch_size=1,
            rank=16,
            mixed_precision=True
        )
        logger.info(
            f"✓ Memory estimation: {
                memory_info['recommended_vram_gb']:.1f} GB recommended")
    except Exception as e:
        logger.error(f"✗ Memory estimation failed: {e}")
        return False

    try:
        target_modules = get_flux_target_modules()
        logger.info(f"✓ Target modules: {len(target_modules)} modules found")
    except Exception as e:
        logger.error(f"✗ Target modules failed: {e}")
        return False

    return True


def test_training_data_preparation():
    """Test training data preparation"""
    logger.info("Testing training data preparation...")

    try:
        test_data = []
        for i in range(3):
            image_b64 = create_test_image(color=(255, i * 50, i * 100))
            test_data.append({
                "image": image_b64,
                "description": f"A test image with color variation {i}"
            })

        logger.info(f"✓ Created {len(test_data)} test samples")
        return test_data

    except Exception as e:
        logger.error(f"✗ Training data preparation failed: {e}")
        return None


def test_model_initialization():
    """Test model initialization without actual training"""
    logger.info("Testing model initialization...")

    try:
        training_module = flux_training_module
        logger.info("✓ Training module initialized")

        logger.info(f"✓ Model path: {training_module.model_path}")
        logger.info(f"✓ Device: {training_module.device}")

        return True

    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("Starting FLUX LoRA training tests...")

    if not test_flux_training_utils():
        logger.error("Utilities test failed")
        return False

    test_data = test_training_data_preparation()
    if test_data is None:
        logger.error("Data preparation test failed")
        return False

    if not test_model_initialization():
        logger.error("Model initialization test failed")
        return False

    logger.info("All tests passed! ✓")

    logger.info("\nExample training call:")
    logger.info("flux_training_module.train_lora(")
    logger.info("    train_data=test_data,")
    logger.info("    user_id=123,")
    logger.info("    rank=16,")
    logger.info("    alpha=16,")
    logger.info("    learning_rate=1e-4,")
    logger.info("    epochs=1,")
    logger.info("    batch_size=1")
    logger.info(")")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
