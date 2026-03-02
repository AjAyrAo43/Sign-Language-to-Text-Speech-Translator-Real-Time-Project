"""
FastAPI backend for the Sign Language to Text/Speech Translator.
Serves the frontend and handles prediction API requests.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from web_app.backend.inference import SignLanguagePredictor
from web_app.backend.gloss_labels import get_all_glosses

# --- App Init ---
app = FastAPI(
    title="Sign Language Translator API",
    description="Real-time ASL sign language to text/speech translation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Init ---
predictor = SignLanguagePredictor()

# --- Serve frontend static files ---
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "web_app", "frontend")
print(f"[INFO] Frontend directory: {FRONTEND_DIR}")
print(f"[INFO] Frontend exists: {os.path.isdir(FRONTEND_DIR)}")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    print(f"[WARNING] Frontend directory not found at {FRONTEND_DIR}")


# --- Schemas ---
class LandmarkFrame(BaseModel):
    """A single frame of landmarks: list of [x, y] pairs."""
    landmarks: List[List[float]]


class PredictionRequest(BaseModel):
    """A sequence of landmark frames for prediction."""
    frames: List[LandmarkFrame]


class PredictionResponse(BaseModel):
    success: bool
    predicted_label: int = -1
    predicted_gloss: str = "N/A"
    confidence: float = 0.0
    top5: list = []
    error: str = ""


# --- Routes ---
@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Sign Language Translator API is running. Frontend not found."}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "num_classes": 100,
    }


@app.get("/api/glosses")
async def get_glosses():
    """Return the full list of supported sign language glosses."""
    glosses = get_all_glosses()
    return {"glosses": glosses, "count": len(glosses)}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict the sign language gloss from a sequence of landmark frames.
    Each frame should contain landmarks as [[x1,y1], [x2,y2], ...].
    """
    if not request.frames:
        raise HTTPException(status_code=400, detail="No frames provided")

    # Convert to the format expected by the predictor
    landmarks_sequence = [frame.landmarks for frame in request.frames]

    result = predictor.predict(landmarks_sequence)

    return PredictionResponse(
        success=result["success"],
        predicted_label=result["predicted_label"],
        predicted_gloss=result["predicted_gloss"],
        confidence=result["confidence"],
        top5=result["top5"],
        error=result.get("error", ""),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
