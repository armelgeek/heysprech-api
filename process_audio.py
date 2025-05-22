#!/usr/bin/env python3
import os
import sys
import redis
from config import *

def ajouter_fichier_queue(redis_client, audio_path):
    """Ajoute un fichier audio à la queue de transcription"""
    if os.path.isfile(audio_path) and audio_path.lower().endswith(AUDIO_EXTENSIONS):
        redis_client.lpush(TRANSCRIPTION_QUEUE, audio_path)
        return True
    return False

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <audio_file_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

    if os.path.isfile(path):
        if ajouter_fichier_queue(redis_client, path):
            print(f"Fichier ajouté à la queue: {path}")
        else:
            print(f"Format non supporté: {path}")
    
    elif os.path.isdir(path):
        count = 0
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            if ajouter_fichier_queue(redis_client, filepath):
                count += 1
                print(f"Fichier ajouté à la queue: {filepath}")
        print(f"\nTotal: {count} fichiers ajoutés à la queue")
    
    else:
        print(f"Chemin invalide: {path}")
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
