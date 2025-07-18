"""
LLM (Large Language Model) module for prompt enhancement
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

class LLMModule:
    """
    Large Language Model module for improving user prompts
    """
    def __init__(self, model_path: str = None):
        """
        Initialize LLM module
        Args:
            model_path: Path to LLM model (defaults to settings.LLM_MODEL_PATH)
        """
        self.model_path = model_path or settings.LLM_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing LLM module with model: {self.model_path}")
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load LLM model and tokenizer"""
        try:
            logger.info("Loading LLM model...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            self.model.eval()
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            raise

    def _create_system_instruction(self) -> str:
        """
        Create system instruction for prompt improvement
        Returns:
            System instruction text
        """
        return """Ты эксперт по созданию текстовых описаний для генерации изображений товаров на маркетплейсах.
Твоя задача - улучшить и конкретизировать пользовательский запрос, сделав его более детальным и подходящим для создания привлекательного изображения товара.
Принципы улучшения:
1. Добавь детали о стиле, освещении, фоне
2. Укажи целевую аудиторию (молодежь, семьи, бизнес и т.д.)
3. Добавь маркетинговые характеристики (современный, экологичный, премиум и т.д.)
4. Сделай описание конкретным и визуально привлекательным
5. Сохрани суть исходного запроса
Отвечай только улучшенным промптом, без дополнительных объяснений."""

    def _tokenize_input(self, text: str) -> dict:
        """
        Tokenize input text
        Args:
            text: Input text to tokenize
        Returns:
            Tokenized input dictionary
        """
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        except Exception as e:
            logger.error(f"Error tokenizing input: {e}")
            raise

    def improve_prompt(self, user_prompt: str) -> str:
        """
        Improve user prompt for image generation
        Implementation of algorithm 7.2.4 from the diploma
        Args:
            user_prompt: Original user prompt
        Returns:
            Improved prompt for image generation
        """
        try:
            if not user_prompt or not isinstance(user_prompt, str):
                logger.warning("Invalid user prompt provided")
                return user_prompt or "Создай изображение товара"
            
            user_prompt = user_prompt.strip()
            if len(user_prompt) == 0:
                return "Создай изображение товара"
            
            if self.model is None:
                self.load_model()

            system_instruction = self._create_system_instruction()
            
            formatted_input = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\nУлучши этот запрос для генерации изображения: {user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = self._tokenize_input(formatted_input)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            improved_prompt = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            improved_prompt = improved_prompt.strip()
            improved_prompt = improved_prompt.replace("<|im_end|>", "").strip()
            
            if not improved_prompt or len(improved_prompt) < 5:
                logger.warning("Generated prompt is too short, using original")
                return user_prompt

            logger.info(f"Original prompt: {user_prompt}")
            logger.info(f"Improved prompt: {improved_prompt[:100]}...")
            return improved_prompt
        except Exception as e:
            logger.error(f"Error improving prompt: {e}")
            return user_prompt

    def batch_improve_prompts(self, prompts: List[str]) -> List[str]:
        """
        Improve multiple prompts
        Args:
            prompts: List of original prompts
        Returns:
            List of improved prompts
        """
        if not prompts:
            return []
        
        improved_prompts = []
        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"Improving prompt {i + 1}/{len(prompts)}")
                improved = self.improve_prompt(prompt)
                improved_prompts.append(improved)
            except Exception as e:
                logger.error(f"Error improving prompt {i + 1}: {e}")
                improved_prompts.append(prompt or "Создай изображение товара")
        return improved_prompts

llm_module = LLMModule()
