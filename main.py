from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from inference_brain import load_model as load_brain_model, predict_image as predict_brain_image
from inference_chest import load_model as load_chest_model, predict_image as predict_chest_image

app = FastAPI(title="Medical Image Classifier (Brain + Chest)")

# Load models
brain_model = load_brain_model("best_brain_tumor_resnet18.pth")
chest_model = load_chest_model("best_chest_model.pth")

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_used: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(model_type: str, file: UploadFile = File(...)):
    # Accept all image types
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")

    try:
        img_bytes = await file.read()

        if model_type == "brain":
            predicted_class, confidence = predict_brain_image(brain_model, img_bytes)
        elif model_type == "chest":
            predicted_class, confidence = predict_chest_image(chest_model, img_bytes)
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type parameter")

        return PredictionResponse(predicted_class=predicted_class, confidence=confidence, model_used=model_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
