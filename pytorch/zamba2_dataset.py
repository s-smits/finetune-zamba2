 import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict

class Zamba2Dataset(Dataset):
    """Custom Dataset for Zamba2 model."""

    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }