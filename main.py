from fastapi import FastAPI, UploadFile, File
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torch
import librosa
import io

app = FastAPI()

# Load the model once when the server starts
model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def get_embedding(audio_bytes):
    # Convert bytes to signal
    signal, fs = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    tensor = torch.tensor(signal).unsqueeze(0)
    return model.encode_batch(tensor)

@app.post("/compare")
async def compare_voices(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Read files
    emb1 = get_embedding(await file1.read())
    emb2 = get_embedding(await file2.read())
    
    # Calculate similarity
    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    return {
        "similarity_score": round(similarity, 4),
        "is_same_person": similarity > 0.80  # Threshold can be adjusted
    }

@app.get("/")
def home():
    return {"status": "Voice Recognition API is running"}
