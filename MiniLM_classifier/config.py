# ============================================================
# GLOBAL CONFIG
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_LENGTH = 128

BATCH_SIZE = 8

LEARNING_RATE = 2e-5

EPOCHS = 5

# ============================================================
# LABELS
# ============================================================

LABELS = {
    "Health": 0,
    "Travel": 1,
    "Invitation": 2,
    "Finance": 3,
    "Visiting_Card": 4
}

ID2LABEL = {v: k for k, v in LABELS.items()}

NUM_CLASSES = len(LABELS)

# ============================================================
# PATHS
# ============================================================

MODEL_SAVE_PATH = "minilm_document_classifier.pth"