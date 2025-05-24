# FLUX LoRA Training Implementation

## Overview

This document describes the implementation of FLUX.1-dev LoRA training based on the ai-toolkit architecture and algorithm 7.2.2 from the diploma work.

## Architecture

### Key Components

1. **FluxLoRATrainingModule**: Main training orchestrator
2. **FluxTrainingDataset**: Dataset with latent and text embedding caching
3. **flux_training_utils**: Utility functions for FLUX-specific operations
4. **Settings**: Configuration management for training parameters

### Based on ai-toolkit

The implementation is based on the proven ai-toolkit architecture with the following key features:

- **Flow Matching**: Uses flow matching scheduler instead of DDPM
- **Latent Caching**: Pre-computes and caches VAE latents for efficiency
- **Text Embedding Caching**: Pre-computes CLIP and T5 embeddings
- **Memory Optimization**: Quantization and gradient checkpointing
- **Guidance Bypass**: Bypasses guidance embedding during training

## Key Features

### 1. Flow Matching Training

```python
# Flow matching forward process
t = timesteps.float() / scheduler_timesteps
noisy_latents = t * latents + (1 - t) * noise
target = latents - noise  # Velocity target
```

### 2. Dual Text Encoder Support

FLUX uses both CLIP and T5 text encoders:

```python
clip_embeds, t5_embeds = encode_prompts_flux(
    tokenizers, text_encoders, prompts
)
combined_embeds = combine_flux_embeddings(clip_embeds, t5_embeds)
```

### 3. Memory Optimization

- **Quantization**: T5 encoder quantized to 8-bit
- **Gradient Checkpointing**: Reduces memory usage during backprop
- **Mixed Precision**: Uses bfloat16 for training
- **Latent Caching**: Avoids repeated VAE encoding

### 4. LoRA Configuration

Targets key FLUX transformer modules:

```python
target_modules = [
    "to_q", "to_k", "to_v", "to_out.0",  # Attention
    "proj_in", "proj_out",               # Projections
    "ff.net.0.proj", "ff.net.2"         # MLP
]
```

## Configuration

### Settings

Key configuration parameters in `settings.py`:

```python
# FLUX LoRA Training Settings
FLUX_LORA_RANK: int = 16
FLUX_LORA_ALPHA: int = 16
FLUX_LEARNING_RATE: float = 1e-4
FLUX_BATCH_SIZE: int = 1
FLUX_EPOCHS: int = 1

# Memory Optimization
QUANTIZE_TEXT_ENCODERS: bool = True
MIXED_PRECISION: bool = True
GRADIENT_CHECKPOINTING: bool = True

# Flow Matching
FLOW_MATCHING_TIMESTEPS: int = 1000
```

### Memory Requirements

Estimated VRAM usage for different configurations:

| Rank | Batch Size | Mixed Precision | Estimated VRAM |
|------|------------|-----------------|----------------|
| 16   | 1          | Yes             | ~26 GB         |
| 32   | 1          | Yes             | ~28 GB         |
| 16   | 2          | Yes             | ~30 GB         |

## Usage

### Basic Training

```python
from src.models.training_module import flux_training_module

# Prepare training data
train_data = [
    {
        "image": base64_encoded_image,
        "description": "A detailed description of the image"
    },
    # ... more samples
]

# Train LoRA
s3_url = flux_training_module.train_lora(
    train_data=train_data,
    user_id=123,
    rank=16,
    alpha=16,
    learning_rate=1e-4,
    epochs=1,
    batch_size=1
)
```

### Advanced Configuration

```python
# Custom training parameters
s3_url = flux_training_module.train_lora(
    train_data=train_data,
    user_id=123,
    rank=32,           # Higher rank for more capacity
    alpha=32,          # Match alpha to rank
    learning_rate=5e-5, # Lower learning rate
    epochs=2,          # More epochs
    batch_size=1       # Keep batch size at 1
)
```

## Training Process

### Algorithm 7.2.2 Implementation

1. **Data Preparation**: Load and cache latents and text embeddings
2. **Model Setup**: Load FLUX components and apply LoRA
3. **Training Loop**: Flow matching training with proper loss computation
4. **Checkpointing**: Save intermediate and final LoRA weights
5. **Upload**: Store trained weights to S3

### Training Loop Details

```python
for epoch in range(epochs):
    for batch in dataloader:
        # Sample timesteps
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        # Add noise
        noise = torch.randn_like(latents)
        
        # Flow matching interpolation
        t = timesteps.float() / 1000
        noisy_latents = t * latents + (1 - t) * noise
        target = latents - noise
        
        # Forward pass
        model_pred = transformer(
            noisy_latents, timesteps, text_embeds
        )
        
        # Compute loss
        loss = F.mse_loss(model_pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

## Utilities

### Memory Estimation

```python
from src.models.flux_training_utils import estimate_flux_memory_usage

memory_info = estimate_flux_memory_usage(
    batch_size=1,
    rank=16,
    mixed_precision=True
)
print(f"Recommended VRAM: {memory_info['recommended_vram_gb']:.1f} GB")
```

### Configuration Validation

```python
from src.models.flux_training_utils import validate_flux_training_config

validate_flux_training_config(
    batch_size=1,
    learning_rate=1e-4,
    rank=16,
    alpha=16
)
```

## Best Practices

### 1. Data Preparation

- Use high-quality images (1024x1024 recommended)
- Provide detailed, accurate captions
- Include diverse scenes and compositions
- 10-20 images typically sufficient for face training

### 2. Training Parameters

- **Rank**: Start with 16, increase to 32 for more capacity
- **Alpha**: Usually equal to rank
- **Learning Rate**: 1e-4 is a good starting point
- **Batch Size**: Keep at 1 due to memory constraints
- **Epochs**: 1 epoch often sufficient

### 3. Memory Management

- Enable mixed precision training
- Use gradient checkpointing
- Quantize text encoders
- Monitor VRAM usage during training

### 4. Quality Control

- Monitor training loss
- Save checkpoints regularly
- Test intermediate results
- Validate on held-out samples

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size to 1
   - Enable all memory optimizations
   - Use smaller LoRA rank

2. **Training Instability**
   - Lower learning rate
   - Check data quality
   - Ensure proper normalization

3. **Poor Results**
   - Increase training data
   - Improve caption quality
   - Adjust LoRA rank/alpha

### Error Messages

- `CUDA out of memory`: Reduce batch size or rank
- `Invalid image format`: Ensure PNG format
- `Text encoding failed`: Check caption length

## Performance Optimization

### Training Speed

- Use latent caching (enabled by default)
- Enable mixed precision
- Use gradient checkpointing
- Optimize data loading

### Memory Usage

- Quantize text encoders
- Use CPU offloading for non-training components
- Clear cache between training runs

## Integration

### API Integration

The training module integrates with the existing API:

```python
# In endpoints.py
from src.models import flux_training_module

@app.post("/train-lora")
async def train_lora(request: TrainLoRARequest):
    s3_url = flux_training_module.train_lora(
        train_data=request.train_data,
        user_id=request.user_id,
        **request.training_params
    )
    return {"model_url": s3_url}
```

### Backward Compatibility

The implementation maintains backward compatibility:

```python
# Legacy import still works
from src.models import training_module  # Points to flux_training_module
```

## Future Improvements

1. **Multi-GPU Support**: Distribute training across multiple GPUs
2. **Advanced Schedulers**: Implement cosine annealing, warmup
3. **Validation Metrics**: Add CLIP score, aesthetic score evaluation
4. **Resume Training**: Support resuming from checkpoints
5. **Hyperparameter Tuning**: Automatic parameter optimization

## References

- [ai-toolkit](https://github.com/ostris/ai-toolkit): Base implementation
- [FLUX.1 Paper](https://arxiv.org/abs/2408.06072): Model architecture
- [LoRA Paper](https://arxiv.org/abs/2106.09685): Low-rank adaptation
- [Flow Matching](https://arxiv.org/abs/2210.02747): Training methodology 