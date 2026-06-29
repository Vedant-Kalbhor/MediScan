# MediScan: AI-Powered Medical Scan Guide

MediScan is a medical imaging classification project that provides an initial automated interpretation of common scans before a clinician review.

## Features
- Brain tumor detection from MRI
- Chest disease classification from CT
- Breast cancer analysis from ultrasound
- Kidney condition detection from CT
- Bone fracture detection from X-ray

## Project Layout
Each model now lives in its own folder with a standard layout:

```text
brain/
  train.py
  inference.py
  best_brain_model.pth

breast/
  train.py
  inference.py
  best_breast_model.pth

chest/
  train.py
  inference.py
  best_chest_model.pth

kidney/
  train.py
  inference.py
  best_kidney_model.pth

bone/
  train.py
  inference.py
  best_bone_model.pth
```

## Datasets
| Scan Type | Source Link |
|-----------|-------------|
| Brain MRI | [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Chest CT | [Kaggle - Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
| Breast Ultrasound | [Kaggle - Breast Ultrasound Images](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| Kidney CT | [Kaggle - CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
| Bone X-ray | [Kaggle - Bone Fracture Detection Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) |

## Tech Stack
- FastAPI
- Streamlit
- PyTorch
- Torchvision
- Pillow
- Requests
- MLflow

## Run
1. Install dependencies:

   ```bash
   pip install fastapi uvicorn streamlit torch torchvision pillow requests python-multipart pydantic
   ```

2. Start the backend:

   ```bash
   python main.py
   ```

3. Start the frontend:

   ```bash
   streamlit run streamlit_app.py
   ```

## Training
Each folder has a `train.py` script and a Kaggle-ready `train.ipynb` notebook that benchmark multiple architectures and save:
- the best model weights as `best_<name>_model.pth`
- the discovered class order as `classes.json`
- benchmark results as `best_<name>_model_benchmark.csv`
- architecture metadata as `best_<name>_model_metadata.json`

The benchmark compares:
- CNN baseline
- ResNet50
- EfficientNetB0
- EfficientNetB3
- ViT
- optional ConvNeXt Tiny

You can override the dataset location with environment variables:
- `BRAIN_DATA_DIR`
- `BREAST_DATA_DIR`
- `CHEST_DATA_DIR`
- `KIDNEY_DATA_DIR`
- `BONE_DATA_DIR`

## Phase 4: MLflow Experiment Tracking
Yes, MLflow fits this project well for the training side.

Use it to track every benchmark run and keep the interview-friendly structure:

```text
MediScan
  Run1
  Run2
  Run3
```

Track these items per run:
- learning rate
- optimizer
- batch size
- accuracy
- F1 score
- confusion matrix

The repo now includes [`mlflow_utils.py`](./mlflow_utils.py), a small helper you can import from any training notebook or future `train.py` script.

Example usage:

```python
from mlflow_utils import start_run, log_classification_report, log_confusion_matrix

with start_run(
    run_name="Run1",
    experiment_name="MediScan",
    tags={"model": "densenet121", "organ": "brain"},
):
    log_classification_report(
        accuracy=0.80,
        f1_score=0.79,
        optimizer="Adam",
        learning_rate=1e-4,
        batch_size=32,
    )
    log_confusion_matrix(y_true, y_pred, class_names)
```

MLflow artifacts are stored locally in `./mlruns/` by default.

## Phase 8: Database Prediction Storage
Yes, PostgreSQL fits this project well for storing inference history and analytics.

The backend now saves every successful prediction into a `prediction_logs` table with:
- `id`
- `timestamp`
- `organ`
- `prediction`
- `confidence`

By default, the app reads `DATABASE_URL`.

Example PostgreSQL URL:

```text
postgresql+psycopg2://mediscan:mediscan@localhost:5432/mediscan
```

If `DATABASE_URL` is not set, the backend falls back to a local SQLite file so the app still runs for demos.

Useful endpoints:
- `POST /predict` stores each prediction automatically
- `GET /predictions?limit=50` returns recent records for analytics dashboards
- `GET /predictions/export?limit=500` downloads CSV for spreadsheet analysis

## Local Data Viewing
You can inspect the stored prediction table locally in a few ways:

1. Streamlit dashboard
   - Run `streamlit run streamlit_app.py`
   - Switch `Workspace` to `Admin Dashboard`
   - View the live table, summary metrics, and CSV download

2. PostgreSQL GUI tools
   - pgAdmin
   - DBeaver
   - TablePlus
   - DataGrip

   Use the connection settings from `DATABASE_URL`, for example:
   - host: `localhost`
   - port: `5432`
   - database: `mediscan`
   - username: `mediscan`
   - password: `mediscan`

3. Terminal / CLI
   - PostgreSQL shell:

   ```bash
   psql "postgresql://mediscan:mediscan@localhost:5432/mediscan"
   ```

   Then run:

   ```sql
   \dt
   SELECT * FROM prediction_logs ORDER BY timestamp DESC LIMIT 20;
   ```

4. Seed script for demo data
   - Run `python scripts/seed_prediction_logs.py`
   - This inserts a small sample set if the table is empty

5. Migration script
   - Run the SQL file in [`sql/001_create_prediction_logs.sql`](./sql/001_create_prediction_logs.sql) against PostgreSQL if you want to create the table manually

## Disclaimer
MediScan is not a medical diagnostic tool. Always consult a licensed medical professional for diagnosis and treatment.
