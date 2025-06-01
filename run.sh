#!/bin/bash

# Vérifie si un fichier audio est fourni en argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <fichier_audio>"
    exit 1
fi

# Création du répertoire sprech-audio dans le home de l'utilisateur
mkdir -p "$HOME/sprech-audio"

# Récupère le chemin relatif du fichier audio par rapport au dossier courant
AUDIO_FILE=$(basename "$1")
AUDIO_DIR=$(dirname "$1")

# Construction de l'image si elle n'existe pas
docker build -t heysprech-api .

# Exécution du conteneur avec le dossier sprech-audio dans le home
docker run -it \
    -v "$(pwd)/$AUDIO_DIR":/workdir \
    -v "$HOME/sprech-audio":/root/sprech-audio \
    heysprech-api "/workdir/$AUDIO_FILE"
