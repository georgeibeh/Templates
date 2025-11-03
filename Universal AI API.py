"""
=========================================================
 UNIVERSAL FASTAPI AI TEMPLATE
=========================================================
A single FastAPI framework that can handle any AI task:
- Text/NLP
- Computer Vision (CV)
- Tabular ML
- Audio / Speech
- Multimodal pipelines

Plug in any model by defining its load, preprocess,
predict, and postprocess functions.
=========================================================
"""

# -------------------------------
# Imports
# -------------------------------
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Union, List, Optional
import joblib
import torch
import os
import io
import numpy as np
from PIL import Image

# -------------------------------
#  App Initialization
# -------------------------------
app = FastAPI(
    title="Universal AI Inference API",
    description="A modular FastAPI framework for any AI model — NLP, CV, or ML.",
    version="1.0.0"
)

# -------------------------------
#  CORS Configuration
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #  restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
#  Directory Setup
# -------------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
#  Model Management
# -------------------------------
#  Load multiple models (CV, NLP, ML, etc.) dynamically

MODELS = {}

def load_model(model_name: str, path: str, framework: str = "torch"):
    """
    Loads a model based on the given framework.
    Supported: 'torch', 'joblib', 'custom'
    """
    if framework == "torch":
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
    elif framework == "joblib":
        model = joblib.load(path)
    elif framework == "custom":
        # Add custom loader logic (e.g. Hugging Face, TensorFlow)
        raise NotImplementedError("Custom loader not yet implemented.")
    else:
        raise ValueError("Unsupported framework.")
    MODELS[model_name] = model
    print(f" Loaded {model_name} model from {path}")

# Example: preload a model (optional)
# load_model("text_classifier", "models/text_model.pkl", framework="joblib")

# -------------------------------
#  Universal Preprocessing
# -------------------------------

def preprocess_input(input_data: Union[str, UploadFile], task_type: str):
    """
    Generic preprocessing dispatcher.
    - For NLP: tokenize text
    - For CV: convert image to tensor
    - For tabular: parse CSV
    """
    if task_type == "text":
        return input_data.strip()
    elif task_type == "image":
        image = Image.open(io.BytesIO(input_data.file.read())).convert("RGB")
        image = image.resize((224, 224))
        tensor = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor
    elif task_type == "csv":
        import pandas as pd
        return pd.read_csv(io.BytesIO(input_data.file.read()))
    else:
        raise HTTPException(status_code=400, detail="Unsupported task type.")


# -------------------------------
#  Universal Prediction
# -------------------------------

def predict(model_name: str, processed_input):
    """
    Handles inference using the appropriate model.
    You can extend this per task type or model.
    """
    model = MODELS.get(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

    try:
        if isinstance(processed_input, str):  # NLP
            output = model.predict([processed_input])[0]
        elif isinstance(processed_input, torch.Tensor):  # CV
            with torch.no_grad():
                output = model(processed_input)
                if isinstance(output, torch.Tensor):
                    output = output.tolist()
        else:  # Tabular or other
            output = model.predict(processed_input)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# -------------------------------
#  Universal Postprocessing
# -------------------------------

def postprocess_output(output, task_type: str):
    """
    Convert model output into user-friendly format.
    """
    if task_type == "text":
        return {"label": output}
    elif task_type == "image":
        return {"output_vector": output}
    elif task_type == "csv":
        return {"predictions": output.tolist() if hasattr(output, "tolist") else output}
    else:
        return {"result": output}


# -------------------------------
#  Routes
# -------------------------------

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """
    Minimal UI for file or text input.
    """
    return """
    <html>
        <head><title>Universal AI API</title></head>
        <body style="font-family:sans-serif;">
            <h2> Universal AI Inference Service</h2>
            <p>Supports Text, Image, and CSV inputs.</p>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input type="text" name="text_input" placeholder="Enter text (optional)"/><br><br>
                <input type="file" name="file" accept=".csv,image/*"/><br><br>
                <select name="task_type">
                    <option value="text">Text</option>
                    <option value="image">Image</option>
                    <option value="csv">CSV</option>
                </select><br><br>
                <input type="text" name="model_name" placeholder="Enter model name (e.g. text_classifier)"/><br><br>
                <input type="submit" value="Run Inference"/>
            </form>
        </body>
    </html>
    """


@app.post("/predict")
async def universal_predict(
    model_name: str = Form(...),
    task_type: str = Form(...),
    text_input: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
     Unified endpoint for all AI prediction tasks.
    Supports:
    - Text input (Form)
    - File input (image, csv, etc.)
    """
    # Validate model
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded.")

    # Prepare input
    input_data = text_input if text_input else file
    if not input_data:
        raise HTTPException(status_code=400, detail="No input provided.")

    # Preprocess → Predict → Postprocess
    processed_input = preprocess_input(input_data, task_type)
    raw_output = predict(model_name, processed_input)
    final_output = postprocess_output(raw_output, task_type)

    return JSONResponse(content={
        "model": model_name,
        "task_type": task_type,
        "result": final_output
    })


# -------------------------------
#  Run the API
# -------------------------------
# Run using: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
