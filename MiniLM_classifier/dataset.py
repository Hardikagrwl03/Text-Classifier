import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class OCRDataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = texts
        self.labels = labels

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }