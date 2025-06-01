FROM python:3.9-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Création des répertoires de travail
WORKDIR /app
RUN mkdir -p /app/de /app/en /app/fr

# Installation des dépendances Python de base
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Copie des fichiers nécessaires
COPY requirements.txt .
COPY cli.py .
COPY opus-mt-de-fr/ ./opus-mt-de-fr/

# Installation des autres dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Point d'entrée
ENTRYPOINT ["python", "cli.py"]
