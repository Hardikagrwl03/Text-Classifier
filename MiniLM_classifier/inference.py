import torch
import time

from transformers import AutoTokenizer

from model import MiniLMClassifier

from config import (
    MODEL_NAME,
    MAX_LENGTH,
    MODEL_SAVE_PATH,
    ID2LABEL
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ============================================================
# LOAD MODEL
# ============================================================

load_start_time = time.time()

model = MiniLMClassifier()

model.load_state_dict(
    torch.load(
        MODEL_SAVE_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

load_end_time = time.time()
load_time_ms = (load_end_time - load_start_time) * 1000
model_size_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)
model_size_mb = model_size_bytes / (1024 * 1024)

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def classify_document(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = (
        encoding["input_ids"].to(device)
    )

    attention_mask = (
        encoding["attention_mask"].to(device)
    )

    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask
        )

        pred = torch.argmax(
            logits,
            dim=1
        ).item()

    return ID2LABEL[pred]

# ============================================================
# TEST
# ============================================================

sample_text = """
Harshil weds Anjali
"""

prediction = classify_document(
    sample_text
)

start_time = time.time()
prediction = classify_document(
    sample_text
)
end_time = time.time()

inference_time_ms = (end_time - start_time) * 1000

print(f"Device: {device}")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Model Load Time: {load_time_ms:.2f} ms")
print(f"Inference Time: {inference_time_ms:.2f} ms")
print("\nPrediction:", prediction)