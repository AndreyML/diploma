"""
FLUX Training Utilities
"""
import torch
import torch.nn.functional as F
from typing import Tuple, List
import logging
logger = logging.getLogger(__name__)


def apply_noise_offset(
        noise: torch.Tensor,
        noise_offset: float) -> torch.Tensor:
    """
    Apply noise offset to improve training stability
    Args:
        noise: Input noise tensor
        noise_offset: Noise offset value
    Returns:
        Modified noise tensor
    """
    if noise_offset is None or (
            noise_offset < 0.000001 and noise_offset > -0.000001):
        return noise
    if len(noise.shape) > 4:
        raise ValueError(
            "Applying noise offset not supported for video models at this time.")
    noise = noise + noise_offset * \
        torch.randn((noise.shape[0], noise.shape[1], 1, 1), device=noise.device)
    return noise


def flow_matching_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    timesteps: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute flow matching loss for FLUX training
    Args:
        model_pred: Model prediction
        target: Target tensor (velocity)
        timesteps: Timestep tensor
        reduction: Loss reduction method
    Returns:
        Computed loss
    """
    loss = F.mse_loss(model_pred, target, reduction=reduction)
    return loss


def prepare_latents_for_flux(
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler_timesteps: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare latents for FLUX flow matching training
    Args:
        latents: Clean latents
        noise: Random noise
        timesteps: Training timesteps
        scheduler_timesteps: Total scheduler timesteps
    Returns:
        Tuple of (noisy_latents, target)
    """
    t = timesteps.float() / scheduler_timesteps
    t = t.view(-1, 1, 1, 1)
    noisy_latents = t * latents + (1 - t) * noise
    target = latents - noise
    return noisy_latents, target


def encode_prompts_flux(
    tokenizers: List,
    text_encoders: List,
    prompts: List[str],
    max_length_clip: int = 77,
    max_length_t5: int = 512,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompts using FLUX text encoders (CLIP + T5)
    Args:
        tokenizers: List of [CLIP tokenizer, T5 tokenizer]
        text_encoders: List of [CLIP encoder, T5 encoder]
        prompts: List of text prompts
        max_length_clip: Max length for CLIP
        max_length_t5: Max length for T5
        device: Target device
    Returns:
        Tuple of (clip_embeds, t5_embeds)
    """
    clip_tokenizer, t5_tokenizer = tokenizers
    clip_encoder, t5_encoder = text_encoders
    clip_inputs = clip_tokenizer(
        prompts,
        max_length=max_length_clip,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        clip_embeds = clip_encoder(**clip_inputs).last_hidden_state
    t5_inputs = t5_tokenizer(
        prompts,
        max_length=max_length_t5,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        t5_embeds = t5_encoder(**t5_inputs).last_hidden_state
    return clip_embeds, t5_embeds


def combine_flux_embeddings(
    clip_embeds: torch.Tensor,
    t5_embeds: torch.Tensor
) -> torch.Tensor:
    """
    Combine CLIP and T5 embeddings for FLUX
    Args:
        clip_embeds: CLIP text embeddings
        t5_embeds: T5 text embeddings
    Returns:
        Combined embeddings
    """
    combined_embeds = torch.cat([clip_embeds, t5_embeds], dim=-1)
    return combined_embeds


def get_flux_target_modules() -> List[str]:
    """
    Get target modules for FLUX LoRA training
    Returns:
        List of target module names
    """
    return [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2"
    ]


def calculate_flux_lora_scale(rank: int, alpha: int) -> float:
    """
    Calculate LoRA scaling factor for FLUX
    Args:
        rank: LoRA rank
        alpha: LoRA alpha
    Returns:
        Scaling factor
    """
    return alpha / rank


def validate_flux_training_config(
    batch_size: int,
    learning_rate: float,
    rank: int,
    alpha: int
) -> bool:
    """
    Validate FLUX training configuration
    Args:
        batch_size: Training batch size
        learning_rate: Learning rate
        rank: LoRA rank
        alpha: LoRA alpha
    Returns:
        True if valid, raises ValueError if not
    """
    if batch_size > 4:
        logger.warning(
            f"Large batch size ({batch_size}) may cause OOM for FLUX training")
    if learning_rate > 1e-3:
        logger.warning(
            f"High learning rate ({learning_rate}) may cause instability")
    if rank > 64:
        logger.warning(
            f"High LoRA rank ({rank}) may slow training significantly")
    if alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {alpha}")
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    return True


def estimate_flux_memory_usage(
    batch_size: int,
    image_size: Tuple[int, int] = (1024, 1024),
    rank: int = 16,
    mixed_precision: bool = True
) -> dict:
    """
    Estimate memory usage for FLUX training
    Args:
        batch_size: Training batch size
        image_size: Image dimensions
        rank: LoRA rank
        mixed_precision: Whether using mixed precision
    Returns:
        Dictionary with memory estimates
    """
    h, w = image_size
    base_model_gb = 24.0
    latent_h, latent_w = h // 8, w // 8
    latent_memory_mb = batch_size * 16 * latent_h * \
        latent_w * (2 if mixed_precision else 4) / (1024 * 1024)
    lora_params_mb = rank * 2 * 100 * 4 / (1024 * 1024)
    gradient_memory_mb = lora_params_mb
    optimizer_memory_mb = lora_params_mb * 2
    total_gb = base_model_gb + \
        (latent_memory_mb + lora_params_mb + gradient_memory_mb + optimizer_memory_mb) / 1024
    return {
        "base_model_gb": base_model_gb,
        "latent_memory_mb": latent_memory_mb,
        "lora_params_mb": lora_params_mb,
        "gradient_memory_mb": gradient_memory_mb,
        "optimizer_memory_mb": optimizer_memory_mb,
        "total_estimated_gb": total_gb,
        "recommended_vram_gb": total_gb * 1.2
    }


def flux_guidance_bypass_forward(self, timestep, guidance, pooled_projection):
    """
    Bypass guidance embedding during training to save memory

    """
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype)
    )
    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = timesteps_emb + pooled_projections
    return conditioning


def bypass_flux_guidance(transformer):
    """
    Bypass FLUX guidance embedding for training
    """
    if hasattr(transformer.time_text_embed, '_bfg_orig_forward'):
        return
    if not hasattr(transformer.time_text_embed, 'guidance_embedder'):
        return
    transformer.time_text_embed._bfg_orig_forward = transformer.time_text_embed.forward
    transformer.time_text_embed.forward = lambda timestep, guidance, pooled_projection: \
        flux_guidance_bypass_forward(transformer.time_text_embed, timestep, guidance, pooled_projection)


def restore_flux_guidance(transformer):
    """
    Restore FLUX guidance embedding after training
    """
    if not hasattr(transformer.time_text_embed, '_bfg_orig_forward'):
        return
    transformer.time_text_embed.forward = transformer.time_text_embed._bfg_orig_forward
    del transformer.time_text_embed._bfg_orig_forward
