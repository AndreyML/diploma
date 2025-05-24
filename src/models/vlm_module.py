"""
VLM (Visual-Language Model) module for image description generation
Implementation based on Qwen2.5-VL-72B-Instruct as specified in the diploma
"""
import base64
import io
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from typing import List, Optional
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

class VLMModule:
    """
    Visual-Language Model module for generating image descriptions
    Based on algorithm 7.2.3 from the diploma work
    """
    def __init__(self, model_path: str = None):
        """
        Initialize VLM module
        Args:
            model_path: Path to VLM model (defaults to settings.VLM_MODEL_PATH)
        """
        self.model_path = model_path or settings.VLM_MODEL_PATH
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing VLM module with model: {self.model_path}")
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load VLM model and processors"""
        try:
            logger.info("Loading VLM model...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.eval()
            logger.info("VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading VLM model: {e}")
            raise

    def _preprocess_image(self, base64_image: str) -> Image.Image:
        """
        Preprocess image from base64 string
        Args:
            base64_image: Base64 encoded image
        Returns:
            PIL Image object
        """
        try:
            if not base64_image or not isinstance(base64_image, str):
                raise ValueError("Invalid base64 image data")
            
            try:
                image_data = base64.b64decode(base64_image)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {e}")
            
            if len(image_data) == 0:
                raise ValueError("Empty image data after decoding")
            
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            max_size = 2048
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.info(f"Image resized to {image.size} to prevent memory issues")
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """
        Tokenize text prompt
        Args:
            prompt: Text prompt for image description
        Returns:
            Tokenized prompt tensor
        """
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Invalid prompt")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {e}")
            raise

    def describe_image(self, base64_image: str, prompt: str = "Опиши, что изображено на картинке") -> str:
        """
        Generate description for a single image
        Implementation of algorithm 7.2.3 from the diploma
        Args:
            base64_image: Base64 encoded image
            prompt: Text prompt for description (default: "Опиши, что изображено на картинке")
        Returns:
            Generated image description
        """
        try:
            if self.model is None:
                self.load_model()

            image = self._preprocess_image(base64_image)
            
            formatted_prompt = f"<|im_start|>system\nТы помощник для описания изображений товаров для маркетплейсов.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = self.processor(
                text=[formatted_prompt],
                images=[image],
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            description = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            description = description.strip()
            if not description:
                return "Не удалось сгенерировать описание изображения"
            
            logger.info(f"Generated description: {description[:100]}...")
            return description
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return f"Ошибка генерации описания: {str(e)}"

    def describe_images(self, base64_images: List[str]) -> List[str]:
        """
        Generate descriptions for multiple images
        Args:
            base64_images: List of base64 encoded images
        Returns:
            List of generated descriptions
        """
        if not base64_images:
            return []
        
        descriptions = []
        for i, base64_image in enumerate(base64_images):
            try:
                logger.info(f"Processing image {i + 1}/{len(base64_images)}")
                description = self.describe_image(base64_image)
                descriptions.append(description)
            except Exception as e:
                logger.error(f"Error processing image {i + 1}: {e}")
                descriptions.append(f"Ошибка обработки изображения: {str(e)}")
        
        return descriptions

vlm_module = VLMModule()
