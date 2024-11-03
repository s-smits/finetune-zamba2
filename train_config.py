import os

HF_TOKEN = os.getenv("HF_TOKEN")

CONFIG = {
    # Model settings
    'model_name': "Zyphra/Zamba2-1.2B-instruct",
    'context_window': 1024,  # has to fit in 4090
    'dataset_name': "BramVanroy/dolly-15k-dutch",
    'split': "train_sft",
    'max_samples': None,
    
    # Training settings
    'num_train_epochs': 2,
    'per_device_train_batch_size': 4,
    'gradient_accumulation_steps': 8,
    'save_steps': 500,
    'save_total_limit': 2,
    'logging_steps': 100,
    'weight_decay': 0.01,
    'fp16': False,
    'bf16': True,
    'dataloader_num_workers': 4,
    'gradient_checkpointing': True,
    'max_grad_norm': 1.0,
    'warmup_steps': 100,
    'output_dir': "./zamba2-finetuned",
    
    # LR optimization settings
    'num_trials': 10,
    'lr_range': (1e-6, 1e-4)
} 