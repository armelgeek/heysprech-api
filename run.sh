#!/bin/bash

# Vérifie si un fichier audio est fourni en argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <fichier_audio>"
    exit 1
fi

# Récupère le chemin absolu du fichier audio
AUDIO_PATH=$(realpath "$1")
AUDIO_DIR=$(dirname "$AUDIO_PATH")
AUDIO_FILE=$(basename "$AUDIO_PATH")

# Construction de l'image si elle n'existe pas
docker build -t heysprech-api .

# Exécution du conteneur
docker run -it \
    -v "$AUDIO_DIR":/workdir \
    -v "$HOME/sprech-audio":/root/sprech-audio \
    heysprech-api "/workdir/$AUDIO_FILE"
