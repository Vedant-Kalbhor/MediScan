from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from inference import load_model, predict_image

app = FastAPI(title="Brain Tumor MRI Classifier")

# Load model at startup
MODEL_PATH = "best_brain_tumor_resnet18.pth"  # adjust path
model = load_model(MODEL_PATH)

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # check file content type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")
    try:
        img_bytes = await file.read()
        class_name, confidence = predict_image(model, img_bytes)
        return PredictionResponse(predicted_class=class_name, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
