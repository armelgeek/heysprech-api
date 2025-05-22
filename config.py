#!/usr/bin/env python3
import os

# Chemins
MODEL_PATH = "./opus-mt-de-fr"
OUTPUT_DIR = "./output"

# Configuration Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Files d'attente Redis
TRANSCRIPTION_QUEUE = "transcription_queue"
TRANSLATION_QUEUE = "translation_queue"

# Extensions
AUDIO_EXTENSIONS = (".opus", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".aiff", ".wma")
EXTENSION_TRANSCRIPTION = ".json"

# Paramètres Whisper
WHISPER_MODEL = "base"
WHISPER_LANGUAGE = "de"
