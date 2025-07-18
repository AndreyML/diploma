"""
FLUX.1-dev image generation module with LoRA support
"""
import torch
import base64
import io
import uuid
from PIL import Image
from diffusers import FluxPipeline
from typing import List, Optional, Tuple
import logging
from src.config.settings import settings
from src.storage.s3_client import s3_client

logger = logging.getLogger(__name__)

class FluxModule:
    """
    FLUX.1-dev image generation module with LoRA support
    """
    def __init__(self, model_path: str = None):
        """
        Initialize FLUX module
        Args:
            model_path: Path to FLUX model (defaults to settings.FLUX_MODEL_PATH)
        """
        self.model_path = model_path or settings.FLUX_MODEL_PATH
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_lora_path = None
        logger.info(f"Initializing FLUX module with model: {self.model_path}")
        logger.info(f"Using device: {self.device}")

    def load_pipeline(self):
        """Load FLUX.1-dev pipeline"""
        try:
            logger.info("Loading FLUX.1-dev pipeline...")
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device == "cuda":
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_sequential_cpu_offload()
            logger.info("FLUX pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FLUX pipeline: {e}")
            raise

    def _tokenize_prompt(self, prompt: str) -> dict:
        """
        Tokenize text prompt for FLUX
        Args:
            prompt: Text prompt for image generation
        Returns:
            Tokenized prompt information
        """
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Invalid prompt")
            return {"prompt": prompt.strip()}
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            raise

    def _load_lora_adapter(self, lora_path: str):
        """
        Load LoRA adapter weights
        Args:
            lora_path: Path to LoRA adapter weights (S3 URL or local path)
        """
        try:
            if self.current_lora_path == lora_path:
                logger.info("LoRA adapter already loaded")
                return

            if lora_path and lora_path.strip():
                logger.info(f"Loading LoRA adapter: {lora_path}")
                
                try:
                    self.pipeline.load_lora_weights(lora_path)
                    self.current_lora_path = lora_path
                    logger.info("LoRA adapter loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load LoRA adapter: {e}")
            else:
                if self.current_lora_path:
                    try:
                        self.pipeline.unload_lora_weights()
                        self.current_lora_path = None
                        logger.info("LoRA adapter unloaded")
                    except Exception as e:
                        logger.warning(f"Could not unload LoRA adapter: {e}")
        except Exception as e:
            logger.error(f"Error handling LoRA adapter: {e}")

    def _generate_image_tensor(self, prompt: str, image_size: Tuple[int, int], guidance_scale: float, num_inference_steps: int) -> torch.Tensor:
        """
        Generate image tensor using diffusion process
        Args:
            prompt: Text prompt
            image_size: Target image size (width, height)
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of diffusion steps
        Returns:
            Generated image tensor
        """
        try:
            width, height = image_size
            
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")
            
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator(device=self.device).manual_seed(42)
            )
            
            if not result.images or len(result.images) == 0:
                raise RuntimeError("No images generated")
            
            return result.images[0]
        except Exception as e:
            logger.error(f"Error generating image tensor: {e}")
            raise

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        Args:
            image: PIL Image object
        Returns:
            Base64 encoded image string
        """
        try:
            if image is None:
                raise ValueError("Image is None")
            
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            
            if len(image_data) == 0:
                raise RuntimeError("Empty image data")
            
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            raise

    def generate_images(
        self,
        prompt: str,
        lora_path: Optional[str] = None,
        num_images: int = 1,
        image_size: Tuple[int, int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ) -> List[str]:
        """
        Generate images from text prompt
        Implementation of algorithm 7.2.1 from the diploma
        Args:
            prompt: Text prompt for image generation
            lora_path: Path to LoRA adapter (optional)
            num_images: Number of images to generate
            image_size: Target image size (width, height)
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of diffusion steps
        Returns:
            List of base64 encoded images
        """
        try:
            if self.pipeline is None:
                self.load_pipeline()

            if not prompt or not isinstance(prompt, str):
                raise ValueError("Invalid prompt provided")

            if num_images <= 0:
                raise ValueError("Number of images must be positive")
            
            if num_images > settings.MAX_IMAGES_PER_REQUEST:
                num_images = settings.MAX_IMAGES_PER_REQUEST
                logger.warning(f"Number of images limited to {settings.MAX_IMAGES_PER_REQUEST}")

            if image_size is None:
                image_size = settings.DEFAULT_IMAGE_SIZE

            tokenized_prompt = self._tokenize_prompt(prompt)

            self._load_lora_adapter(lora_path)

            generated_images = []
            for i in range(num_images):
                logger.info(f"Generating image {i + 1}/{num_images}")
                
                image = self._generate_image_tensor(
                    prompt=prompt,
                    image_size=image_size,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )

                base64_image = self._image_to_base64(image)
                generated_images.append(base64_image)
                logger.info(f"Image {i + 1} generated successfully")

            logger.info(f"Generated {len(generated_images)} images for prompt: {prompt[:50]}...")
            return generated_images
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            raise

    def generate_and_save_images(
        self,
        prompt: str,
        user_id: int,
        lora_path: Optional[str] = None,
        num_images: int = 1
    ) -> Tuple[List[str], List[str]]:
        """
        Generate images and save them to S3
        Args:
            prompt: Text prompt for image generation
            user_id: User ID for S3 path organization
            lora_path: Path to LoRA adapter (optional)
            num_images: Number of images to generate
        Returns:
            Tuple of (base64_images, s3_urls)
        """
        try:
            base64_images = self.generate_images(
                prompt=prompt,
                lora_path=lora_path,
                num_images=num_images
            )

            s3_urls = []
            for i, base64_image in enumerate(base64_images):
                try:
                    s3_key = f"users/{user_id}/generated_images/gen_{uuid.uuid4()}.png"
                    image_data = base64.b64decode(base64_image)
                    s3_url = s3_client.upload_image(image_data, s3_key)
                    s3_urls.append(s3_url)
                    logger.info(f"Image {i + 1} saved to S3: {s3_url}")
                except Exception as e:
                    logger.warning(f"Failed to save image {i + 1} to S3: {e}")
                    s3_urls.append("")

            return base64_images, s3_urls
        except Exception as e:
            logger.error(f"Error generating and saving images: {e}")
            raise

flux_module = FluxModule()
