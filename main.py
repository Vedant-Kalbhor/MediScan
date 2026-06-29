from datetime import datetime
import csv
import io
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
from model_manager import manager
from config import MODELS_CONFIG
from database import init_db, log_prediction, fetch_recent_predictions
from monitoring import metrics_response, start_background_metrics, track_request

app = FastAPI(
    title="MediScan API",
    description="Backend API for medical image classification using deep learning models.",
    version="2.0.0"
)

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_used: str
    scan_name: str


class PredictionRecord(BaseModel):
    id: int
    timestamp: datetime
    organ: str
    prediction: str
    confidence: float


@app.on_event("startup")
def on_startup():
    init_db()
    start_background_metrics()


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start
        track_request(request, status_code, duration)

@app.get("/models")
async def list_models():
    """Returns a list of available models and their descriptions."""
    return {k: v["name"] for k, v in MODELS_CONFIG.items()}


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return metrics_response()

@app.post("/predict", response_model=PredictionResponse)
async def predict(background_tasks: BackgroundTasks, model_type: str, file: UploadFile = File(...)):
    if model_type not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Choose from: {list(MODELS_CONFIG.keys())}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")

    try:
        img_bytes = await file.read()
        predicted_class, confidence = manager.predict(model_type, img_bytes)

        if predicted_class == "Model not found/loaded":
            raise HTTPException(status_code=503, detail="Model weights not found on server. Please train the model.")

        background_tasks.add_task(
            log_prediction,
            organ=model_type,
            prediction=predicted_class,
            confidence=confidence,
        )

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            model_used=model_type,
            scan_name=MODELS_CONFIG[model_type]["name"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/predictions", response_model=list[PredictionRecord])
def list_predictions(limit: int = Query(50, ge=1, le=500)):
    """Return recent stored predictions for analytics and dashboards."""
    rows = fetch_recent_predictions(limit=limit)
    return [
        {
            "id": row.id,
            "timestamp": row.timestamp,
            "organ": row.organ,
            "prediction": row.prediction,
            "confidence": row.confidence,
        }
        for row in rows
    ]


@app.get("/predictions/export")
def export_predictions_csv(limit: int = Query(500, ge=1, le=5000)):
    """Export recent predictions as CSV for analytics or spreadsheet work."""
    rows = fetch_recent_predictions(limit=limit)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "timestamp", "organ", "prediction", "confidence"])
    for row in rows:
        writer.writerow([
            row.id,
            row.timestamp.isoformat() if row.timestamp else "",
            row.organ,
            row.prediction,
            row.confidence,
        ])

    buffer.seek(0)
    headers = {
        "Content-Disposition": 'attachment; filename="mediscan_predictions.csv"'
    }
    return Response(content=buffer.getvalue(), media_type="text/csv", headers=headers)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
