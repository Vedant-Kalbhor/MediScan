import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from medical_training_utils import train_image_folder_classifier

DATA_DIR = os.environ.get(
    "BONE_DATA_DIR",
    "/kaggle/input/bone-fracture-detection-computer-vision-project",
)
MODEL_SAVE_NAME = Path(__file__).with_name("best_bone_model.pth")
CLASS_NAMES_OUTPUT = Path(__file__).with_name("classes.json")


def main():
    train_image_folder_classifier(
        data_dir=DATA_DIR,
        model_save_name=MODEL_SAVE_NAME,
        class_names_output=CLASS_NAMES_OUTPUT,
    )


if __name__ == "__main__":
    main()
