#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import time
from config import *

def start_process(script_name, args=None):
    """Démarre un script Python comme un processus"""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    return subprocess.Popen(cmd)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <audio_file_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"Le chemin {input_path} n'existe pas.")
        sys.exit(1)

    # Créer les répertoires nécessaires
    os.makedirs("output", exist_ok=True)

    # Démarrer les workers
    processes = []
    
    # Démarrer le worker de transcription
    print("Démarrage du worker de transcription...")
    transcription_worker = start_process("transcribe_api.py")
    processes.append(transcription_worker)

    # Démarrer le worker de traduction
    print("Démarrage du worker de traduction...")
    translation_worker = start_process("traduction_lignes.py")
    processes.append(translation_worker)

    # Attendre que les workers soient prêts
    time.sleep(2)

    # Ajouter les fichiers à la queue
    print("Ajout des fichiers à la queue...")
    process_audio = start_process("process_audio.py", [input_path])
    processes.append(process_audio)

    def signal_handler(signum, frame):
        print("\nArrêt des processus...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    # Capturer Ctrl+C pour arrêter proprement
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Attendre que le processus d'ajout de fichiers se termine
        process_audio.wait()
        
        print("\nTous les fichiers ont été ajoutés à la queue.")
        print("Les workers continuent de traiter les fichiers...")
        print("Appuyez sur Ctrl+C pour arrêter le pipeline")
        
        # Maintenir les workers en vie
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nArrêt des processus...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
