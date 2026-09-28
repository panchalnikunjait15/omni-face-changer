from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI(
    title="OmniFace AI Studio API",
    description="World No.1 Face Changer Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "success", "message": "OmniFace AI Backend is live and running!"}

@app.post("/api/v1/swap-face")
async def swap_face(source_photo: UploadFile = File(...), target_video: UploadFile = File(...)):
    try:
        os.makedirs("temp", exist_ok=True)
        photo_path = f"temp/{source_photo.filename}"
        video_path = f"temp/{target_video.filename}"

        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(source_photo.file, buffer)
            
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(target_video.file, buffer)
        
        return {
            "status": "success",
            "message": "Face swap processing initiated!",
            "source_photo": source_photo.filename,
            "target_video": target_video.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))