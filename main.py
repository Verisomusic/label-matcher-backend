from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import tempfile
import os

app = FastAPI()

# Allow your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_track(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"Saved temp file at {tmp_path}")

        # Load audio at 22,050 Hz
        audio, sr = librosa.load(tmp_path, sr=22050)
        print(f"Loaded audio: {len(audio)} samples")

        # Beat detection
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        print(f"Detected tempo: {tempo}")

        # Spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))

        # Clean up
        os.remove(tmp_path)

        return {
            "tempo": float(tempo),
            "centroid": spectral_centroid,
            "bandwidth": spectral_bandwidth
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}
