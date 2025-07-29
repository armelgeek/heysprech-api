#!/bin/bash

# Vérifie si un fichier audio est fourni en argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <fichier_audio>"
    exit 1
fi


# Utilise le chemin du dossier du script pour tous les dossiers
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/audios"
mkdir -p "$SCRIPT_DIR/de"
mkdir -p "$SCRIPT_DIR/fr"
mkdir -p "$SCRIPT_DIR/en"

# Si un fichier est fourni, le copier dans le dossier audios s'il n'y est pas déjà
if [ ! -f "$SCRIPT_DIR/audios/$(basename "$1")" ]; then
    cp "$1" "$SCRIPT_DIR/audios/"
fi

# Construction de l'image si elle n'existe pas


# Utilise le chemin du dossier du script pour éviter les problèmes de droits
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker run -it \
    --volume "$SCRIPT_DIR/audios":/app/audios:ro \
    --volume "$SCRIPT_DIR/de":/app/de:rw \
    --volume "$SCRIPT_DIR/fr":/app/fr:rw \
    --volume "$SCRIPT_DIR/en":/app/en:rw \
    heysprech-api "/app/audios/$(basename "$1")" --source-lang de --target-lang fr