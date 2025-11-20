from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import tempfile
import traceback

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")

        # Save uploaded file temporarily
        suffix = ".wav" if file.filename.lower().endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        print(f"Saved temporary file at: {tmp_path}")

        # Load audio with standard sample rate
        audio, sr = librosa.load(tmp_path, sr=22050, mono=True)
        print(f"Loaded audio: {len(audio)} samples at {sr} Hz")

        # Analyze tempo
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        print(f"Analysis complete: tempo={tempo}")

        # Dummy labels (replace with your matching logic later)
        matching_labels = [
            {"label": "Anjunadeep", "score": 0.92},
            {"label": "Monstercat", "score": 0.89},
        ]

        # Dummy similar tracks
        similar_tracks = [
            {"artist": "Artist 1", "title": "Track A", "label": "Label X"},
            {"artist": "Artist 2", "title": "Track B", "label": "Label Y"},
        ]

        return {
            "tempo": float(tempo),
            "labels": matching_labels,
            "tracks": similar_tracks
        }

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return {"error": str(e)}
