# MediScan: AI-Powered Medical Scan Guide

MediScan is a medical imaging classification project that provides an initial automated interpretation of common scans before a clinician review.

## Features
- Brain tumor detection from MRI
- Chest disease classification from CT
- Breast cancer analysis from ultrasound
- Kidney condition detection from CT
- Bone fracture detection from X-ray with fracture localization

## Project Layout
Each model lives in its own folder with a standard layout:

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
  best_bone_model.pt
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
- SQLAlchemy
- Hugging Face Hub
- Docker for deployment
- Render for hosting

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

The local workflow stays exactly the same. Docker is only needed for deployment or optional container-based testing.

## Training
Each folder has a `train.py` script and a Kaggle-ready `train.ipynb` notebook that benchmark multiple architectures and save:
- the best model weights as `best_<name>_model.pth`
- the discovered class order as `classes.json`
- benchmark results as `best_<name>_model_benchmark.csv`
- architecture metadata as `best_<name>_model_metadata.json`

For the bone model, the backend also returns YOLO bounding boxes and a human-readable image region so the Streamlit UI can highlight where the fracture appears without changing the brain, breast, chest, or kidney models.

The benchmark compares:
- CNN baseline
- ResNet50
- DenseNet
- MobileNetV3

You can override the dataset location with environment variables:
- `BRAIN_DATA_DIR`
- `BREAST_DATA_DIR`
- `CHEST_DATA_DIR`
- `KIDNEY_DATA_DIR`
- `BONE_DATA_DIR`

## Database Storage
The backend stores every successful prediction in a `prediction_logs` table with:
- `id`
- `timestamp`
- `organ`
- `prediction`
- `confidence`

By default, the app uses local SQLite at `sqlite:///./mediscan.db`.
If you set `DATABASE_URL`, it switches to PostgreSQL automatically.

Useful endpoints:
- `POST /predict` stores each prediction automatically
- `GET /predictions?limit=50` returns recent records for analytics dashboards
- `GET /predictions/export?limit=500` downloads CSV for spreadsheet analysis

## Deployment Architecture

```text
My Laptop
    |
    v
GitHub
    |
    v
Render
    |
    +--> FastAPI backend starts from Dockerfile.backend
    |
    +--> First prediction request loads a model on demand
    |
    +--> Hugging Face provides model weights if local files are absent
    |
    +--> The most recently used model is cached in memory
    |
    +--> Inference results are stored in SQLite or PostgreSQL
    |
    +--> Streamlit frontend starts from Dockerfile.frontend
```

## Hugging Face Setup

The deployment loader expects your model repo to be:

`vedk08/MediScan-Models`

Suggested files in that repo:

- `best_brain_model.pth`
- `best_breast_model.pth`
- `best_chest_model.pth`
- `best_kidney_model.pth`
- `best_bone_model.pt`

Environment variables used by the deployment loader:

- `HF_MODEL_REPO` points to the Hugging Face repo
- `HF_TOKEN` is only needed if the repo is private
- `MODEL_CACHE_DIR` controls where downloaded weights are cached
- `MAX_LOADED_MODELS` caps how many models stay resident in RAM at once; the Render blueprint sets this to `1`

## Docker Usage

Backend image:

```bash
docker build -f Dockerfile.backend -t mediscan-backend .
```

Frontend image:

```bash
docker build -f Dockerfile.frontend -t mediscan-frontend .
```

## docker-compose Usage

To run both services in containers locally:

```bash
docker compose up --build
```

Then open:

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

The compose file keeps the backend database in a mounted volume and points the frontend at the backend service name.

## Render Deployment Steps

You can deploy with the included [`render.yaml`](./render.yaml) blueprint or manually from the Render dashboard.

Blueprint flow:

1. Connect the repository to Render.
2. Apply `render.yaml`.
3. Provide `HF_TOKEN` if your Hugging Face repo is private.
4. Let Render create the PostgreSQL database and the two web services.
5. The backend service runs on a `standard` Render plan so Torch inference has more room than the smallest tier.

Manual flow:

1. Create a backend Web Service from `Dockerfile.backend`.
2. Set `HF_MODEL_REPO=vedk08/MediScan-Models`.
3. Add `HF_TOKEN` if required.
4. Point `DATABASE_URL` to your Render PostgreSQL connection string.
5. Create a frontend Web Service from `Dockerfile.frontend`.
6. Set `BACKEND_HOSTPORT` or `BACKEND_URL` so Streamlit can call FastAPI.
7. Keep `MAX_LOADED_MODELS=1` on the backend if you want to cap RAM usage as tightly as possible.

For a step-by-step version, see [`deployment/render.md`](./deployment/render.md).

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
