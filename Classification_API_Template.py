"""
=========================================================
 FASTAPI MACHINE LEARNING TEMPLATE (No Data Processing)
=========================================================
A minimal, flexible FastAPI app for handling CSV uploads,
ML model prediction, and returning results.

 Designed for quick adaptation to any ML project.
=========================================================
"""

# -------------------------------
#  Imports
# -------------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# -------------------------------
#  App Initialization
# -------------------------------
app = FastAPI(
    title="FastAPI ML Prediction Template",
    description="Minimal FastAPI app for CSV upload and model inference.",
    version="1.0.0"
)

# -------------------------------
#  CORS Configuration (Optional)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #  Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
#  Directory Setup
# -------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "models"  # Recommended folder for your ML models

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# -------------------------------
#  Load Trained Model
# -------------------------------
# Replace with your own model file and path
MODEL_FILENAME = os.path.join(MODEL_PATH, "covid_model.pkl")

try:
    model = joblib.load(MODEL_FILENAME)
except Exception as e:
    raise RuntimeError(f" Failed to load model: {MODEL_FILENAME}\nError: {e}")

# -------------------------------
#  Prediction Function
# -------------------------------

def make_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
     Run model predictions on the uploaded DataFrame.

    This is a placeholder — modify to match your model’s expected input.
    Example: you might select specific columns before prediction.
    """
    try:
        predictions = model.predict(df)
    except Exception as e:
        raise ValueError(f"Model prediction failed: {e}")
    
    df["Prediction"] = predictions
    return df


# -------------------------------
# Routes
# -------------------------------

@app.get("/", response_class=HTMLResponse)
async def homepage():
    """
    Simple HTML upload form.
    Replace or remove if integrating with a frontend.
    """
    return """
    <html>
        <head><title>ML Prediction API</title></head>
        <body style="font-family:sans-serif;">
            <h2> FastAPI ML Prediction Service</h2>
            <p>Upload a CSV file to generate predictions.</p>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept=".csv"/>
                <input type="submit" value="Upload & Predict"/>
            </form>
        </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
     Upload a CSV file and receive a prediction CSV in response.
    """
    # Validate file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        #  Placeholder: apply your own preprocessing here if needed
        # e.g., df = preprocess_data(df)

        # Run model predictions
        result_df = make_predictions(df)

        # Save output file
        output_path = os.path.join(OUTPUT_FOLDER, "predictions.csv")
        result_df.to_csv(output_path, index=False)

        # Return CSV file to user
        return FileResponse(
            path=output_path,
            filename="predictions.csv",
            media_type="text/csv"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# -------------------------------
#  Run Locally
# -------------------------------
# Run with:  uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
