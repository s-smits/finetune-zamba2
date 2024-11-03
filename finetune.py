import os
from lr_optimizer import train_with_lr_optimization
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
from huggingface_hub import login
from training_config import CONFIG, HF_TOKEN

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = kwargs.get('model')
        
    def _move_model_to_device(self, model, device):
        pass  # model already mapped to devices

def main():
    # Login to Hugging Face if token provided
    if HF_TOKEN:
        login(HF_TOKEN)

    try:
        # Get training setup with LR optimization
        training_setup = train_with_lr_optimization(**CONFIG)
        
        if training_setup:
            # Initialize custom trainer
            trainer = CustomTrainer(
                model=training_setup['model'],
                args=training_setup['training_args'],
                train_dataset=training_setup['dataset'],
                data_collator=training_setup['data_collator'],
                callbacks=training_setup['callbacks']
            )
            
            # Train the model
            trainer.train()
            
            # Save the final model and tokenizer
            training_setup['model'].save_pretrained(CONFIG['output_dir'] + "-final")
            training_setup['tokenizer'].save_pretrained(CONFIG['output_dir'] + "-final")
            
            print(f"Training completed successfully! Model saved to {CONFIG['output_dir']}-final")
            
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()