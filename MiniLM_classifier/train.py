import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader

from dataset import OCRDataset
from model import MiniLMClassifier

from config import (
    LABELS,
    ID2LABEL,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_SAVE_PATH
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/dataset.csv")

df["label_id"] = df["label"].map(LABELS)

train_texts, val_texts, train_labels, val_labels = (
    train_test_split(
        df["text"].tolist(),
        df["label_id"].tolist(),
        test_size=0.2,
        random_state=42
    )
)

# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = OCRDataset(
    train_texts,
    train_labels
)

val_dataset = OCRDataset(
    val_texts,
    val_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# MODEL
# ============================================================

model = MiniLMClassifier()

model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids = (
            batch["input_ids"].to(device)
        )

        attention_mask = (
            batch["attention_mask"].to(device)
        )

        labels = (
            batch["label"].to(device)
        )

        optimizer.zero_grad()

        logits = model(
            input_ids,
            attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch+1} "
        f"Loss: {avg_loss:.4f}"
    )

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print("\nModel Saved!")

# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in val_loader:

        input_ids = (
            batch["input_ids"].to(device)
        )

        attention_mask = (
            batch["attention_mask"].to(device)
        )

        labels = (
            batch["label"].to(device)
        )

        logits = model(
            input_ids,
            attention_mask
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=list(LABELS.keys())
    )
)