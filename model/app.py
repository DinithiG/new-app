from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import numpy as np
import io
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any

from models.lcnn_model import LCNNModel
from models.spectrogram_model import SpectrogramModel
from models.hybrid_model import HybridModel
from utils.audio_processing import preprocess_audio

app = FastAPI()

# Configure CORS to allow requests from your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
lcnn_model = LCNNModel().to(device)
spectrogram_model = SpectrogramModel().to(device)
hybrid_model = HybridModel(lcnn_model, spectrogram_model).to(device)

# Load pre-trained weights (you'll need to train these models first)
try:
    lcnn_model.load_state_dict(torch.load("models/weights/lcnn_weights.pth", map_location=device))
    spectrogram_model.load_state_dict(torch.load("models/weights/spectrogram_weights.pth", map_location=device))
    print("Models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model weights: {e}")
    print("Using untrained models for demonstration")

# Set models to evaluation mode
lcnn_model.eval()
spectrogram_model.eval()
hybrid_model.eval()

class AnalysisResult(BaseModel):
    score: float
    probability: str
    details: List[Dict[str, Any]]

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    # Read and process the audio file
    contents = await file.read()
    audio_bytes = io.BytesIO(contents)
    
    # Preprocess audio
    waveform, spectrogram = preprocess_audio(audio_bytes, device)
    
    # Process with hybrid model
    with torch.no_grad():
        prediction, feature_scores = hybrid_model(waveform, spectrogram)
        
    # Convert prediction to float
    score = prediction.item()
    
    # Determine probability text
    probability = "Low"
    if score > 0.7:
        probability = "High"
    elif score > 0.4:
        probability = "Medium"
    
    # Create detailed analysis from feature scores
    details = [
        {"name": "Frequency Patterns", "score": feature_scores[0].item()},
        {"name": "Voice Naturalness", "score": feature_scores[1].item()},
        {"name": "Background Noise", "score": feature_scores[2].item()},
        {"name": "Temporal Consistency", "score": feature_scores[3].item()},
        {"name": "Spectral Artifacts", "score": feature_scores[4].item()},
    ]
    
    return {
        "score": score,
        "probability": probability,
        "details": details
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)