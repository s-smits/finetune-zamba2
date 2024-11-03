CONTEXT_WINDOW = 1024 #has to fit in 4090
HF_TOKEN = "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"
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling
)
import torch
from datasets import load_dataset
from huggingface_hub import login

# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-1.2B-instruct", token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # better for inference

# init model with auto device mapping
model = AutoModelForCausalLM.from_pretrained(
    "Zyphra/Zamba2-1.2B-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"  # handles multi-gpu/cpu mapping
)
model.config.pad_token_id = tokenizer.pad_token_id

# Load the Dutch Dolly dataset
dataset = load_dataset("BramVanroy/dolly-15k-dutch", split="train_sft")

def prepare_chat_format(examples):
    chats = []
    for messages in examples['messages']:
        try:
            chat = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                max_length=CONTEXT_WINDOW,
                truncation=True,
                return_tensors=None
            )
        except Exception as e:
            print(f"Error applying chat template: {e}")
            # Fallback format if chat template fails
            text = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                text += f"<|{role}|>\n{content}</s>\n"
            
            chat = tokenizer(
                text,
                max_length=CONTEXT_WINDOW,
                truncation=True,
                return_tensors=None
            )["input_ids"]
            
        chats.append(chat)
    return {"input_ids": chats}

# Process the dataset
tokenized_dataset = dataset.map(
    prepare_chat_format,
    batched=True,
    remove_columns=dataset.column_names
)

# training config
training_args = TrainingArguments(
    output_dir="./zamba2-finetuned",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    gradient_accumulation_steps=8,
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    warmup_steps=100
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# custom trainer to handle device mapping
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        
    def _move_model_to_device(self, model, device):
        pass  # model already mapped to devices

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Add explicit training and saving steps
trainer.train()
model.save_pretrained("./zamba2-finetuned-final")
tokenizer.save_pretrained("./zamba2-finetuned-final")
