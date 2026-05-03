import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# ============================================================
# LABELS
# ============================================================

LABELS = {
    "Health": 0,
    "Travel": 1,
    "Invitation": 2,
    "Finance": 3,
    "Visiting Card": 4
}

ID2LABEL = {v: k for k, v in LABELS.items()}

# ============================================================
# SAMPLE DATA
# Replace with your OCR dataset
# ============================================================

data = [
    ("Apollo Hospital Prescription for Blood Test", "Health"),
    ("Boarding Pass Delhi to Mumbai Gate 14", "Travel"),
    ("You are cordially invited to the wedding ceremony", "Invitation"),
    ("Account Statement Credit Debit Balance", "Finance"),
    ("John Doe Software Engineer Phone Email", "Visiting Card"),
]

df = pd.DataFrame(data, columns=["text", "label"])

df["label_id"] = df["label"].map(LABELS)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42
)

# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# DATASET
# ============================================================

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
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = OCRDataset(train_texts, train_labels)
val_dataset = OCRDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ============================================================
# CLASSIFIER MODEL
# ============================================================

class MiniLMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        return torch.sum(
            token_embeddings * input_mask_expanded,
            1
        ) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embeddings = self.mean_pooling(outputs, attention_mask)

        logits = self.classifier(embeddings)

        return logits

# ============================================================
# TRAINING SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MiniLMClassifier(num_classes=len(LABELS))
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# ============================================================
# TRAINING LOOP
# ============================================================

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=list(LABELS.keys())
    )
)

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def classify_document(text):

    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        logits = model(input_ids, attention_mask)

        pred = torch.argmax(logits, dim=1).item()

    return ID2LABEL[pred]

# ============================================================
# TEST
# ============================================================

sample_text = """
ICICI Bank Account Statement
Transaction History
Available Balance
"""

prediction = classify_document(sample_text)

print("\nPrediction:", prediction)