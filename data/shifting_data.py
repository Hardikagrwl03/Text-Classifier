from pathlib import Path
import shutil

Labels = ["Utility", "SalarySlip", "ITR_Form16", "BankStatement", "Check"]

for label in Labels:
    source = Path(f"/home/hamster/Downloads/{label}")
    destination = Path(f"./dataset/Finance")

    destination.mkdir(exist_ok=True)

    append_str = label.replace("_", "")

    for img in source.iterdir():
        if img.is_file():
            new_name = f"{img.stem}{append_str}{img.suffix}"
            shutil.move(str(img), destination / new_name)

    print("Done.")