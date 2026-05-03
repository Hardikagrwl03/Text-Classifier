import torch
import torch.nn as nn

from transformers import AutoModel

from config import MODEL_NAME, NUM_CLASSES

class MiniLMClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES)
        )

    def mean_pooling(
        self,
        model_output,
        attention_mask
    ):

        token_embeddings = (
            model_output.last_hidden_state
        )

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            1
        ) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = self.mean_pooling(
            outputs,
            attention_mask
        )

        logits = self.classifier(
            embeddings
        )

        return logits