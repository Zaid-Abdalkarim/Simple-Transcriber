from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import shutil
import os
import time
import mimetypes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# cpu bc its runs on a mini pc
# change base to tiny if its too slow, small is better than base but slower obv.
model = WhisperModel("base", device="cpu")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    mime_type, _ = mimetypes.guess_type(file.filename)
    if not mime_type or not (mime_type.startswith("audio") or mime_type.startswith("video")):
        return {"error": "File must be audio or video."}

    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcript = ""
    transcript_file = filepath + ".txt"
    try:
        segments, _ = model.transcribe(filepath)
        transcript = " ".join([segment.text for segment in segments])

        with open(transcript_file, "w") as f:
            f.write(transcript)

        return {
            "filename": filename,
            "transcript": transcript,
            "transcript_file": os.path.basename(transcript_file)
        }
    finally:
        # Dont want to store files at all
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(transcript_file):
            os.remove(transcript_file)