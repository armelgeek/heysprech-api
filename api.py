#!/usr/bin/env python3
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from database import Database
import redis
from config import *

app = FastAPI()
db = Database()

# Configuration de yt-dlp
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': os.path.join(AUDIO_DIR, '%(id)s.%(ext)s'),
}

class VideoRequest(BaseModel):
    youtube_id: str

@app.post("/api/videos")
async def add_video(video: VideoRequest):
    try:
        # Vérifier si le dossier audio existe
        os.makedirs(AUDIO_DIR, exist_ok=True)
        
        # Télécharger l'audio de la vidéo YouTube
        video_url = f"https://www.youtube.com/watch?v={video.youtube_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Chemin du fichier audio
        audio_path = os.path.join(AUDIO_DIR, f"{video.youtube_id}.mp3")
        
        # Ajouter l'entrée dans la base de données
        video_id = db.add_video(video.youtube_id, audio_path)
        
        # Ajouter le fichier à la queue de traitement
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        redis_client.lpush(TRANSCRIPTION_QUEUE, audio_path)
        
        return {
            "status": "success",
            "message": "Vidéo ajoutée avec succès",
            "video_id": video_id,
            "youtube_id": video.youtube_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}")
async def get_video(video_id: int):
    try:
        video_data = db.get_video(video_id)
        if not video_data:
            raise HTTPException(status_code=404, detail="Vidéo non trouvée")
        return video_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos")
async def list_videos():
    """Liste toutes les vidéos"""
    try:
        videos = db.get_all_videos()
        return videos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def system_status():
    """Obtient l'état du système"""
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        return {
            "transcription_queue": redis_client.llen(TRANSCRIPTION_QUEUE),
            "translation_queue": redis_client.llen(TRANSLATION_QUEUE),
            "active_workers": count_active_workers()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: int):
    """Supprime une vidéo et ses fichiers associés"""
    try:
        video = db.get_video(video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Vidéo non trouvée")
            
        # Supprimer les fichiers
        if os.path.exists(video["audio_path"]):
            os.remove(video["audio_path"])
            
        # Supprimer de la base de données
        db.delete_video(video_id)
        return {"status": "success", "message": "Vidéo supprimée"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
