# create_dataset.py

# ============================================================
# GENERIC OCR DATASET CREATOR
# ============================================================
#
# This script:
# 1. Reads documents/images from folders
# 2. Runs OCR
# 3. Stores extracted text into CSV
#
# Folder Structure:
#
# dataset/
# ├── Health/
# ├── Travel/
# ├── Invitation/
# ├── Finance/
# └── Visiting_Card/
#
# Output:
# data/dataset.csv
#
# ============================================================

import os
import argparse
import cv2
import pandas as pd
import pytesseract

from tqdm import tqdm
from pdf2image import convert_from_path

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "dataset"

SUPPORTED_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

SUPPORTED_PDF_EXTENSIONS = [
    ".pdf"
]

# ============================================================
# OCR CONFIG
# ============================================================

TESSERACT_CONFIG = r"""
--oem 3
--psm 6
"""

# ============================================================
# IMAGE OCR
# ============================================================

def extract_text_from_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return ""

    # Convert BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    text = pytesseract.image_to_string(
        image,
        config=TESSERACT_CONFIG
    )

    return text.strip()

# ============================================================
# PDF OCR
# ============================================================

def extract_text_from_pdf(pdf_path):

    full_text = ""

    try:

        pages = convert_from_path(pdf_path)

        for page in pages:

            page = cv2.cvtColor(
                np.array(page),
                cv2.COLOR_RGB2BGR
            )

            text = pytesseract.image_to_string(
                page,
                config=TESSERACT_CONFIG
            )

            full_text += "\n" + text

    except Exception as e:

        print(f"PDF OCR Failed: {pdf_path}")
        print(e)

    return full_text.strip()

# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()

# ============================================================
# MAIN
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Extract OCR data to CSV."
    )

    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Only extract this label folder."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    labels = os.listdir(DATASET_DIR)

    if args.label:
        if args.label not in labels:
            print(f"Label not found: {args.label}")
            return

        labels = [args.label]

    for label in labels:
        
        dataset = []

        label_path = os.path.join(
            DATASET_DIR,
            label
        )

        if not os.path.isdir(label_path):
            continue

        print(f"\nProcessing Label: {label}")

        files = os.listdir(label_path)

        for file_name in tqdm(files):

            file_path = os.path.join(
                label_path,
                file_name
            )

            extension = os.path.splitext(
                file_name
            )[1].lower()

            text = ""

            # ====================================================
            # IMAGE OCR
            # ====================================================

            if extension in SUPPORTED_IMAGE_EXTENSIONS:

                text = extract_text_from_image(
                    file_path
                )

            # ====================================================
            # PDF OCR
            # ====================================================

            elif extension in SUPPORTED_PDF_EXTENSIONS:

                text = extract_text_from_pdf(
                    file_path
                )

            else:
                continue

            text = clean_text(text)

            # Skip empty OCR
            if len(text) < 5:
                continue

            dataset.append({
                "text": text,
                "label": label
            })

        # ============================================================
        # SAVE CSV
        # ============================================================

        df = pd.DataFrame(dataset)

        os.makedirs(f"extracted_data/{label}", exist_ok=True)

        df.to_csv(
            f"extracted_data/{label}/dataset.csv",
            index=False
        )

        print("\n================================================")
        print(f"Dataset Saved: extracted_data/{label}/dataset.csv")
        print(f"Total Samples: {len(df)}")
        print("================================================")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()