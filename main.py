from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from model_manager import manager
from config import MODELS_CONFIG

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

@app.get("/models")
async def list_models():
    """Returns a list of available models and their descriptions."""
    return {k: v["name"] for k, v in MODELS_CONFIG.items()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(model_type: str, file: UploadFile = File(...)):
    if model_type not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail=f"Invalid model_type. Choose from: {list(MODELS_CONFIG.keys())}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")

    try:
        img_bytes = await file.read()
        predicted_class, confidence = manager.predict(model_type, img_bytes)

        if predicted_class == "Model not found/loaded":
            raise HTTPException(status_code=503, detail="Model weights not found on server. Please train the model.")

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
