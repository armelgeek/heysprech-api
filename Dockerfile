FROM python:3.9-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Création des répertoires de travail
WORKDIR /workdir
RUN mkdir -p /workdir/de /workdir/en /workdir/fr

# Installation des dépendances Python de base
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Copie des fichiers nécessaires dans /app (pas dans /workdir)
COPY requirements.txt /app/
COPY cli.py /app/
COPY opus-mt-de-fr/ /app/opus-mt-de-fr/

# Installation des autres dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Point d'entrée
ENTRYPOINT ["python", "/app/cli.py"]