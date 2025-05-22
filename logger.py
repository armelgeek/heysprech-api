#!/usr/bin/env python3
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file='app.log', level=logging.INFO):
    """Configure un logger avec rotation des fichiers"""
    
    # Créer le dossier logs s'il n'existe pas
    os.makedirs('logs', exist_ok=True)
    
    # Configurer le logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Gestionnaire de fichier avec rotation
    handler = RotatingFileHandler(
        os.path.join('logs', log_file),
        maxBytes=10000000,  # 10MB
        backupCount=5
    )
    
    # Format du log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Ajouter le gestionnaire au logger
    logger.addHandler(handler)
    
    return logger

# Loggers spécifiques
api_logger = setup_logger('api', 'api.log')
transcription_logger = setup_logger('transcription', 'transcription.log')
translation_logger = setup_logger('translation', 'translation.log')
