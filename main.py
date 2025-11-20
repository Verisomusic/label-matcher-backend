import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydub import AudioSegment
import librosa
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import tempfile

app = FastAPI()

# Spotify auth
sp = Spotify(auth_manager=SpotifyClientCredentials(
    client_id="YOUR_SPOTIFY_CLIENT_ID",
    client_secret="YOUR_SPOTIFY_CLIENT_SECRET"
))

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=44100)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    key = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1).argmax()
    return tempo, int(key)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Extract basic audio features
    tempo, key = extract_features(tmp_path)

    # Search Spotify for similar tracks using audio features
    recos = sp.recommendations(
        limit=20,
        seed_genres=None,
        min_tempo=tempo - 5,
        max_tempo=tempo + 5,
    )

    # Collect label info
    labels = {}
    similar_tracks = []

    for track in recos["tracks"]:
        # Track info
        artist = track["artists"][0]["name"]
        title = track["name"]
        album = sp.album(track["album"]["id"])
        label = album.get("label", "Unknown")

        similar_tracks.append({
            "artist": artist,
            "title": title,
            "label": label
        })

        labels[label] = labels.get(label, 0) + 1

    # Sort labels by count
    sorted_labels = sorted(
        [{"label": l, "count": c} for l, c in labels.items()],
        key=lambda x: x["count"],
        reverse=True
    )

    return {
        "labels": sorted_labels[:10],
        "tracks": similar_tracks
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
