# finetune-zamba2

This repository contains tools and scripts for fine-tuning Zamba2-1.2B-Instruct with a specific dataset. The implementation supports both single and multi-GPU training with automatic learning rate optimization.

## Requirements

- NVIDIA GPU with at least 24GB VRAM (e.g., RTX 4090)
- CUDA 11.8
- Python 3.10+
- Docker (optional)

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/s-smits/finetune-zamba2.git
cd finetune-zamba2
```

2. Set your Hugging Face token:
```bash
export HF_TOKEN="your_token_here"
```

3. Start training:
```bash
docker-compose up --build
```

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/s-smits/finetune-zamba2.git
cd finetune-zamba2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the fine-tuning script:
```bash
python finetune.py
```

## Configuration

The training configuration can be customized in `finetune_config.py`:

```python
CONFIG = {
    'model_name': "Zyphra/Zamba2-1.2B-instruct",
    'context_window': 1024,
    'batch_size': 4,
    'gradient_accumulation_steps': 8,
    'num_train_epochs': 1,
    # ... other parameters
}
```

### Key Parameters

- `context_window`: Maximum sequence length (default: 1024)
- `batch_size`: Training batch size per device
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate (automatically optimized)
- `weight_decay`: Weight decay for AdamW optimizer
- `fp16`/`bf16`: Mixed precision training options

## Features

- 🚀 Automatic learning rate optimization using Optuna
- 🎯 BFloat16 training support for memory efficiency
- 📊 Gradient checkpointing for reduced memory usage
- 🔄 Automatic device mapping for multi-GPU setups
- 🛡️ Robust error handling and logging
- 🐳 Docker support for reproducible environments

## Hardware Requirements

- **Minimum**: GPU with 24GB VRAM (for context window = 1024)
- **Recommended**: GPU with >24GB VRAM (for context window = 2048)
- **Multi-GPU**: Supported through automatic device mapping

## Training Data

The default configuration uses the "BramVanroy/dolly-15k-dutch" dataset. To use your own data:

1. Modify the `dataset_name` in `finetune_config.py`
2. Or prepare your data in the following format:
```python
dataset = Dataset.from_dict({
    "text": [
        "Your training example 1",
        "Your training example 2",
        # ...
    ]
})
```

## Output

The fine-tuned model will be saved in two locations:
- Training checkpoints: `./zamba2-finetuned/`
- Final model: `./zamba2-finetuned-final/`

## Monitoring

Training progress can be monitored through:
- Console output with training metrics
- Tensorboard logs (if enabled in config)
- Weights & Biases integration (if enabled in config)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Reduce context window size

2. **Training Too Slow**
   - Increase batch size
   - Adjust learning rate range
   - Enable mixed precision training

3. **Docker Issues**
   - Ensure NVIDIA Container Toolkit is installed
   - Check CUDA compatibility
   - Verify GPU visibility in container

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

- Hugging Face
- Zamba2 model creators
