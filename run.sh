#!/bin/bash

# Vérifie si un fichier audio est fourni en argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <fichier_audio>"
    exit 1
fi

# Création des dossiers s'ils n'existent pas
mkdir -p "$(pwd)/audios"
mkdir -p "$(pwd)/pronunciations"
mkdir -p "$(pwd)/transcriptions"

# Si un fichier est fourni, le copier dans le dossier audios s'il n'y est pas déjà
if [ ! -f "$(pwd)/audios/$(basename "$1")" ]; then
    cp "$1" "$(pwd)/audios/"
fi

# Construction de l'image si elle n'existe pas
docker build -t heysprech-api .

# Exécution du conteneur avec les volumes montés
docker run -it \
    --volume "$(pwd)/audios":/app/audios:ro \
    --volume "$(pwd)/pronunciations":/app/pronunciations:rw \
    --volume "$(pwd)/transcriptions":/app/transcriptions:rw \
    heysprech-api "/app/audios/$(basename "$1")"
