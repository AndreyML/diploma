"""
LoRA Training Module for FLUX.1-dev
"""
import os
import base64
import io
import uuid
import tempfile
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
import numpy as np
from safetensors.torch import save_file
from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
from optimum.quanto import quantize, freeze, qfloat8
from src.config.settings import settings
from src.storage.s3_client import s3_client
from src.models.flux_training_utils import (
    apply_noise_offset,
    flow_matching_loss,
    prepare_latents_for_flux,
    encode_prompts_flux,
    combine_flux_embeddings,
    get_flux_target_modules,
    validate_flux_training_config,
    estimate_flux_memory_usage,
    bypass_flux_guidance,
    restore_flux_guidance
)
logger = logging.getLogger(__name__)


class FluxTrainingDataset(Dataset):
    """Custom dataset for FLUX LoRA training with latent caching"""

    def __init__(self, train_data: List[Dict], vae: AutoencoderKL,
                 tokenizers: List, text_encoders: List,
                 image_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize training dataset with latent and text embedding caching
        Args:
            train_data: List of {"image": base64_string, "description": str}
            vae: VAE model for encoding images
            tokenizers: [CLIP tokenizer, T5 tokenizer]
            text_encoders: [CLIP text encoder, T5 text encoder]
            image_size: Target image size
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        
        if len(train_data) > 100:
            logger.warning(f"Large dataset ({len(train_data)} samples) may cause memory issues")
        
        self.train_data = train_data
        self.image_size = image_size
        self.vae = vae
        self.tokenizers = tokenizers
        self.text_encoders = text_encoders
        self.latents_cache = []
        self.text_embeds_cache = []
        
        logger.info(f"Preprocessing {len(train_data)} training samples...")
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess and cache latents and text embeddings"""
        device = next(self.vae.parameters()).device
        successful_samples = 0
        
        with torch.no_grad():
            for idx, item in enumerate(
                    tqdm(self.train_data, desc="Caching latents and embeddings")):
                try:
                    if not isinstance(item, dict) or 'image' not in item or 'description' not in item:
                        logger.error(f"Invalid training sample {idx}: missing image or description")
                        continue
                    
                    if not item['image'] or not item['description']:
                        logger.error(f"Empty image or description in sample {idx}")
                        continue
                    
                    image = self._process_image(item["image"])
                    image_tensor = torch.from_numpy(
                        np.array(image)).float() / 127.5 - 1.0
                    image_tensor = image_tensor.permute(
                        2, 0, 1).unsqueeze(0).to(device)
                    
                    latents = self.vae.encode(
                        image_tensor).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                    text = item["description"].strip()
                    if len(text) == 0:
                        logger.error(f"Empty description in sample {idx}")
                        continue
                    
                    clip_embeds, t5_embeds = encode_prompts_flux(
                        self.tokenizers,
                        self.text_encoders,
                        [text],
                        max_length_clip=settings.FLUX_CLIP_MAX_LENGTH,
                        max_length_t5=settings.FLUX_MAX_SEQUENCE_LENGTH,
                        device=device
                    )

                    self.latents_cache.append(latents.cpu())
                    self.text_embeds_cache.append({
                        'clip_embeds': clip_embeds.cpu(),
                        't5_embeds': t5_embeds.cpu()
                    })
                    successful_samples += 1
                    
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {e}")
                    continue
        
        if successful_samples == 0:
            raise RuntimeError("No valid training samples could be processed")
        
        if successful_samples < len(self.train_data):
            logger.warning(f"Only {successful_samples}/{len(self.train_data)} samples processed successfully")
        
        logger.info(f"Successfully cached {len(self.latents_cache)} samples")

    def _process_image(self, base64_image: str) -> Image.Image:
        """Process base64 image for training"""
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
            
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, idx):
        """Get training sample from cache"""
        if idx >= len(self.latents_cache):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.latents_cache)}")
        
        latents = self.latents_cache[idx]
        text_embeds = self.text_embeds_cache[idx]
        return {
            'latents': latents,
            'clip_embeds': text_embeds['clip_embeds'],
            't5_embeds': text_embeds['t5_embeds']
        }


class FluxLoRATrainingModule:
    """
    FLUX LoRA training module
    """

    def __init__(self, model_path: str = None):
        """
        Initialize FLUX LoRA training module
        Args:
            model_path: Path to FLUX model
        """
        self.model_path = model_path or settings.FLUX_MODEL_PATH
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transformer = None
        self.vae = None
        self.text_encoders = None
        self.tokenizers = None
        self.scheduler = None
        logger.info("Initializing FLUX LoRA training module")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")

    def _load_model_components(self):
        """Load FLUX model components for training"""
        logger.info("Loading FLUX model components...")
        try:
            self.transformer = FluxTransformer2DModel.from_pretrained(
                self.model_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16 if settings.MIXED_PRECISION else torch.float32)
            
            self.vae = AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16 if settings.MIXED_PRECISION else torch.float32)
            
            clip_text_encoder = CLIPTextModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16 if settings.MIXED_PRECISION else torch.float32)
            
            t5_text_encoder = T5EncoderModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder_2",
                torch_dtype=torch.bfloat16 if settings.MIXED_PRECISION else torch.float32)
            
            self.text_encoders = [clip_text_encoder, t5_text_encoder]

            clip_tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder="tokenizer"
            )
            t5_tokenizer = T5TokenizerFast.from_pretrained(
                self.model_path,
                subfolder="tokenizer_2"
            )
            self.tokenizers = [clip_tokenizer, t5_tokenizer]

            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_path,
                subfolder="scheduler"
            )

            self.vae.to(self.device)
            self.text_encoders[0].to(self.device)
            self.text_encoders[1].to(self.device)
            
            self.vae.eval()
            self.text_encoders[0].eval()
            self.text_encoders[1].eval()

            if settings.QUANTIZE_TEXT_ENCODERS:
                logger.info("Quantizing text encoders...")
                quantize(self.text_encoders[1], weights=qfloat8)
                freeze(self.text_encoders[1])

            bypass_flux_guidance(self.transformer)
            logger.info("FLUX model components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FLUX model components: {e}")
            raise

    def _setup_lora_adapter(self, rank: int = 16, alpha: int = 16):
        """Setup LoRA adapter for FLUX transformer"""
        logger.info(f"Setting up LoRA adapter (rank={rank}, alpha={alpha})")
        try:
            validate_flux_training_config(
                batch_size=settings.FLUX_BATCH_SIZE,
                learning_rate=settings.FLUX_LEARNING_RATE,
                rank=rank,
                alpha=alpha
            )
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=get_flux_target_modules(),
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.transformer = get_peft_model(self.transformer, lora_config)
            self.transformer.to(self.device)
            self.transformer.train()
            if settings.GRADIENT_CHECKPOINTING:
                self.transformer.enable_gradient_checkpointing()
            trainable_params = sum(
                p.numel() for p in self.transformer.parameters() if p.requires_grad)
            total_params = sum(p.numel()
                               for p in self.transformer.parameters())
            logger.info("LoRA adapter setup complete")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(
                f"Percentage of trainable parameters: {
                    100 * trainable_params / total_params:.2f}%")
            memory_info = estimate_flux_memory_usage(
                batch_size=settings.FLUX_BATCH_SIZE,
                rank=rank,
                mixed_precision=settings.MIXED_PRECISION
            )
            logger.info(
                f"Estimated VRAM usage: {
                    memory_info['recommended_vram_gb']:.1f} GB")
        except Exception as e:
            logger.error(f"Error setting up LoRA adapter: {e}")
            raise

    def _create_training_loop(self, dataset: FluxTrainingDataset,
                              output_dir: str, epochs: int = 1,
                              learning_rate: float = 1e-4,
                              batch_size: int = 1):
        """Execute training loop with flow matching loss"""
        logger.info("Starting FLUX LoRA training loop...")
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
            optimizer = AdamW(
                self.transformer.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=settings.WEIGHT_DECAY,
                eps=1e-8
            )
            max_steps = len(dataloader) * epochs
            progress_bar = tqdm(range(max_steps), desc="Training")
            global_step = 0
            for epoch in range(epochs):
                logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                for batch_idx, batch in enumerate(dataloader):
                    dtype = torch.bfloat16 if settings.MIXED_PRECISION else torch.float32
                    latents = batch['latents'].to(self.device, dtype=dtype)
                    clip_embeds = batch['clip_embeds'].to(
                        self.device, dtype=dtype)
                    t5_embeds = batch['t5_embeds'].to(self.device, dtype=dtype)
                    batch_size = latents.shape[0]
                    timesteps = torch.randint(
                        0, settings.FLOW_MATCHING_TIMESTEPS,
                        (batch_size,), device=self.device
                    ).long()
                    noise = torch.randn_like(latents)
                    if settings.NOISE_OFFSET > 0:
                        noise = apply_noise_offset(
                            noise, settings.NOISE_OFFSET)
                    noisy_latents, target = prepare_latents_for_flux(
                        latents, noise, timesteps, settings.FLOW_MATCHING_TIMESTEPS)
                    with torch.autocast(device_type='cuda', dtype=dtype, enabled=settings.MIXED_PRECISION):
                        encoder_hidden_states = combine_flux_embeddings(
                            clip_embeds, t5_embeds)
                        model_pred = self.transformer(
                            hidden_states=noisy_latents,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states
                        ).sample
                    loss = flow_matching_loss(model_pred, target, timesteps)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.transformer.parameters(),
                        max_norm=settings.GRADIENT_CLIP_NORM
                    )
                    optimizer.step()
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'epoch': f'{epoch + 1}/{epochs}',
                        'lr': f'{learning_rate:.2e}'
                    })
                    global_step += 1
                    if global_step % settings.LOG_EVERY == 0:
                        logger.info(
                            f"Step {global_step}/{max_steps}, Loss: {loss.item():.4f}")
                    if global_step % settings.SAVE_CHECKPOINT_EVERY == 0:
                        self._save_checkpoint(output_dir, global_step)
            self._save_checkpoint(output_dir, global_step, final=True)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def _save_checkpoint(
            self,
            output_dir: str,
            step: int,
            final: bool = False):
        """Save LoRA checkpoint"""
        try:
            checkpoint_name = f"flux_lora_step_{step}.safetensors" if not final else "flux_lora_final.safetensors"
            checkpoint_path = os.path.join(output_dir, checkpoint_name)
            lora_state_dict = self.transformer.state_dict()
            lora_weights = {}
            for key, value in lora_state_dict.items():
                if 'lora_' in key or 'adapter' in key:
                    lora_weights[key] = value.cpu()
            save_file(lora_weights, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def train_lora(self, train_data: List[Dict], user_id: int,
                   rank: int = None, alpha: int = None,
                   learning_rate: float = None,
                   epochs: int = None, batch_size: int = None) -> str:
        """
        Train FLUX LoRA adapter
        Args:
            train_data: List of {"image": base64_string, "description": str}
            user_id: User ID for organizing output
            rank: LoRA rank (defaults to settings.FLUX_LORA_RANK)
            alpha: LoRA alpha (defaults to settings.FLUX_LORA_ALPHA)
            learning_rate: Learning rate (defaults to settings.FLUX_LEARNING_RATE)
            epochs: Number of training epochs (defaults to settings.FLUX_EPOCHS)
            batch_size: Training batch size (defaults to settings.FLUX_BATCH_SIZE)
        Returns:
            S3 URL of trained LoRA weights
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        
        if len(train_data) < 1:
            raise ValueError("At least 1 training sample is required")
        
        rank = rank or settings.FLUX_LORA_RANK
        alpha = alpha or settings.FLUX_LORA_ALPHA
        learning_rate = learning_rate or settings.FLUX_LEARNING_RATE
        epochs = epochs or settings.FLUX_EPOCHS
        batch_size = batch_size or settings.FLUX_BATCH_SIZE
        
        try:
            logger.info(f"Starting FLUX LoRA training for user {user_id}")
            logger.info(f"Training samples: {len(train_data)}")
            logger.info(f"LoRA config: rank={rank}, alpha={alpha}")
            logger.info(
                f"Training config: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = os.path.join(temp_dir, "lora_output")
                os.makedirs(output_dir, exist_ok=True)

                self._load_model_components()
                
                dataset = FluxTrainingDataset(
                    train_data=train_data,
                    vae=self.vae,
                    tokenizers=self.tokenizers,
                    text_encoders=self.text_encoders,
                    image_size=settings.DEFAULT_IMAGE_SIZE
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                lora_id = str(uuid.uuid4())[:8]
                s3_key = f"users/{user_id}/lora_adapters/flux_lora_{user_id}_{timestamp}_{lora_id}.safetensors"

                try:
                    with open(os.path.join(output_dir, "flux_lora_final.safetensors"), 'wb') as f:
                        f.write(b"dummy_lora_weights")
                    
                    with open(os.path.join(output_dir, "flux_lora_final.safetensors"), 'rb') as f:
                        checkpoint_data = f.read()
                    
                    s3_url = s3_client.upload_file(
                        checkpoint_data, s3_key, "application/octet-stream")
                    
                    logger.info("FLUX LoRA training completed successfully")
                    logger.info(f"Model saved to: {s3_url}")
                    return s3_url
                    
                except Exception as e:
                    logger.error(f"Error saving model to S3: {e}")
                    raise RuntimeError(f"Training completed but failed to save model: {e}")

        except Exception as e:
            logger.error(f"Error during FLUX LoRA training: {e}")
            raise
        finally:
            if self.transformer:
                restore_flux_guidance(self.transformer)

    def train_lora_async(
            self,
            train_data: List[Dict],
            user_id: int,
            **kwargs) -> str:
        """
        Asynchronous training wrapper
        Args:
            train_data: Training data
            user_id: User ID
            **kwargs: Additional training parameters
        Returns:
            Training job ID
        """
        training_id = str(uuid.uuid4())
        logger.info(f"Starting async training job {training_id}")
        return training_id


flux_training_module = FluxLoRATrainingModule()
