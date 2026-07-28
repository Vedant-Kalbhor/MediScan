# Render Deployment Guide

This repo keeps the local workflow unchanged:

- Backend: `python main.py`
- Frontend: `streamlit run streamlit_app.py`

For deployment, Render builds the Docker images from:

- [`Dockerfile.backend`](../Dockerfile.backend)
- [`Dockerfile.frontend`](../Dockerfile.frontend)

## Architecture

```text
Laptop / GitHub
    |
    v
Render builds backend image
    |
    v
FastAPI starts
    |
    v
First prediction request triggers model load
    |
    v
Model file is resolved locally or downloaded from Hugging Face
    |
    v
Only the most recent model stays cached in memory
    |
    v
Prediction is written to SQLite or PostgreSQL
```

## Hugging Face Setup

This project expects the model repository to be:

`vedk08/MediScan-Models`

Suggested filenames inside the repo:

- `best_brain_model.pth`
- `best_breast_model.pth`
- `best_chest_model.pth`
- `best_kidney_model.pth`
- `best_bone_model.pt`

If the repository is private, add an `HF_TOKEN` secret in Render.

## Render Blueprint

You can deploy the whole stack with [`render.yaml`](../render.yaml).

It provisions:

- a backend web service
- a frontend web service
- a PostgreSQL / SQLite database

The frontend uses `BACKEND_HOSTPORT` so it can talk to the backend on Render's private network without hardcoding a public URL.

## Manual Render Steps

If you prefer the dashboard instead of the blueprint:

1. Create a new Web Service for the backend.
2. Point it at [`Dockerfile.backend`](../Dockerfile.backend).
3. Add environment variables:
   - `HF_MODEL_REPO=vedk08/MediScan-Models`
   - `HF_TOKEN` if the repo is private
   - `MODEL_CACHE_DIR=/tmp/models`
   - `DATABASE_URL` from a Render PostgreSQL instance
4. Create a second Web Service for the frontend.
5. Point it at [`Dockerfile.frontend`](../Dockerfile.frontend).
6. Add `BACKEND_HOSTPORT` or `BACKEND_URL` so Streamlit can reach FastAPI.

## Docker Compose

For local container testing, use:

```bash
docker compose up --build
```

Services:

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

The compose file uses SQLite in a mounted volume for the backend database.

## Notes

- The backend loads each model on first use and keeps only the most recently used model cached in memory to stay within hosted memory limits.
- Local runs still use the checked-in model files when they are present.
- Docker is only needed for container deployments or compose-based testing.
