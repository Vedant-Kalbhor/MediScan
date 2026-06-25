import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from medical_training_utils import train_image_folder_classifier

DATA_DIR = os.environ.get(
    "BREAST_DATA_DIR",
    "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT",
)
MODEL_SAVE_NAME = Path(__file__).with_name("best_breast_model.pth")
CLASS_NAMES_OUTPUT = Path(__file__).with_name("classes.json")
BENCHMARK_RESULTS_OUTPUT = Path(__file__).with_name("best_breast_model_benchmark.csv")
METADATA_OUTPUT = Path(__file__).with_name("best_breast_model_metadata.json")
EPOCHS = int(os.getenv("EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))


def main():
    train_image_folder_classifier(
        data_dir=DATA_DIR,
        model_save_name=MODEL_SAVE_NAME,
        class_names_output=CLASS_NAMES_OUTPUT,
        benchmark_results_output=BENCHMARK_RESULTS_OUTPUT,
        metadata_output=METADATA_OUTPUT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        include_optional=True,
    )


if __name__ == "__main__":
    main()
