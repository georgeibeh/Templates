"""
=========================================================
 FASTAPI COMPUTER VISION TEMPLATE
=========================================================
A versatile FastAPI template for computer vision projects:
handles image upload, model inference, and result response.

 Works for classification, detection, segmentation, etc.
=========================================================
"""

# -------------------------------
#  Imports
# -------------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
import joblib
import torch
from PIL import Image
import io
import os

# -------------------------------
#  App Initialization
# -------------------------------
app = FastAPI(
    title="FastAPI Computer Vision Template",
    description="Minimal FastAPI app for image upload and model inference.",
    version="1.0.0"
)

# -------------------------------
#  CORS Configuration
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #  Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
#  Directory Setup
# -------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# -------------------------------
#  Load Pretrained Model
# -------------------------------
#  Replace this with your own model loading logic (PyTorch, TensorFlow, etc.)
MODEL_FILENAME = os.path.join(MODEL_PATH, "vision_model.pt")

try:
    model = torch.load(MODEL_FILENAME, map_location=torch.device("cpu"))
    model.eval()  # set model to evaluation mode
except Exception as e:
    raise RuntimeError(f"⚠️ Failed to load model: {MODEL_FILENAME}\nError: {e}")

# -------------------------------
#  Inference Utility
# -------------------------------

def prepare_image(file: UploadFile) -> np.ndarray:
    """
     Convert uploaded file into a NumPy or Torch tensor suitable for inference.
    Modify this function based on your model’s input requirements.
    """
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        # Example preprocessing — adjust as needed:
        image = image.resize((224, 224))
        image_np = np.array(image) / 255.0
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def make_prediction(image_tensor: torch.Tensor):
    """
     Run model inference.
    Modify this to match your specific task (classification, detection, etc.).
    """
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
        # Example for classification models:
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        return {"class_id": pred_class, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


# -------------------------------
#  Routes
# -------------------------------

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """
    Simple HTML upload interface.
    """
    return """
    <html>
        <head><title>Computer Vision API</title></head>
        <body style="font-family:sans-serif;">
            <h2> FastAPI Computer Vision Service</h2>
            <p>Upload an image to get predictions.</p>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="image/*"/>
                <input type="submit" value="Upload & Predict"/>
            </form>
        </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
     Upload an image and receive model prediction.
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only image files (.png, .jpg, .jpeg) are supported.")

    # Save file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Reset file pointer for reading again (PIL needs it)
    file.file.seek(0)

    try:
        # Preprocess image
        image_tensor = prepare_image(file)

        # Run inference
        result = make_prediction(image_tensor)

        # (Optional) Save annotated image or result file if your model supports it
        # Example: save visualization to OUTPUT_FOLDER

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# -------------------------------
#  Run Locally
# -------------------------------
# Run using:  uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
