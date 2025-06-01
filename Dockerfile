FROM python:3.9-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Création des répertoires de travail
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt .
COPY cli.py .
COPY opus-mt-de-fr/ ./opus-mt-de-fr/

# Installation des dépendances Python
RUN pip install -r requirements.txt

# Point d'entrée
ENTRYPOINT ["python", "cli.py"]
