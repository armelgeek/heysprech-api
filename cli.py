#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Configuration simplifiée
CONFIG = {
    'whisper_model': "base",
    'language': "de",
    'output_format': "json",
    'translation_models': {
        'de-fr': "Helsinki-NLP/opus-mt-de-fr",
        'en-fr': "Helsinki-NLP/opus-mt-en-fr",
        'fr-en': "Helsinki-NLP/opus-mt-fr-en",
        'de-en': "Helsinki-NLP/opus-mt-de-en"
    },
    'max_vocabulary_size': 50,
    'min_word_length': 3
}

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")

def get_output_folder(audio_path: str) -> Path:
    """Crée le dossier de sortie pour un fichier audio"""
    audio_name = Path(audio_path).stem
    output_dir = Path.home() / "transcriptions" / audio_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

class SimpleTranslator:
    """Traducteur simple utilisant MarianMT"""
    
    def __init__(self, lang_pair: str):
        self.lang_pair = lang_pair
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle de traduction"""
        if self.lang_pair not in CONFIG['translation_models']:
            raise ValueError(f"Paire de langues non supportée: {self.lang_pair}")
        
        model_name = CONFIG['translation_models'][self.lang_pair]
        print(f"Chargement du modèle de traduction: {model_name}")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def translate(self, text: str) -> str:
        """Traduit un texte"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=3,
                temperature=0.3
            )
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation.strip()
        except Exception as e:
            print(f"Erreur de traduction pour '{text}': {e}")
            return text

class VocabularyExtractor:
    """Extracteur de vocabulaire simplifié"""
    
    def __init__(self, translator: SimpleTranslator, source_lang: str, target_lang: str):
        self.translator = translator
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.processed_words = set()
    
    def clean_word(self, word: str) -> Optional[str]:
        """Nettoie et valide un mot"""
        if not word:
            return None
        
        # Nettoyer le mot
        word = word.strip().lower()
        word = re.sub(r'[^\w\säöüß]', '', word)
        
        # Valider le mot
        if (len(word) < CONFIG['min_word_length'] or 
            len(word) > 15 or 
            not re.match(r'^[a-zA-ZäöüßÄÖÜàâäéèêëïîôùûüÿç]+$', word) or
            word in self.processed_words):
            return None
        
        self.processed_words.add(word)
        return word
    
    def extract_vocabulary(self, segments: List[Dict]) -> List[Dict]:
        """Extrait le vocabulaire des segments"""
        vocabulary = set()
        
        # Extraire les mots uniques
        for segment in segments:
            text = segment.get('text', '').strip()
            words = text.split()
            
            for word in words:
                clean_word = self.clean_word(word)
                if clean_word and len(vocabulary) < CONFIG['max_vocabulary_size']:
                    vocabulary.add(clean_word)
        
        # Créer les entrées de vocabulaire avec traductions
        vocab_entries = []
        print(f"Traduction de {len(vocabulary)} mots...")
        
        for word in tqdm(sorted(vocabulary), desc="Vocabulaire"):
            try:
                translation = self.translator.translate(word)
                vocab_entries.append({
                    'word': word,
                    'translation': translation,
                    'source_lang': self.source_lang,
                    'target_lang': self.target_lang
                })
            except Exception as e:
                print(f"Erreur pour '{word}': {e}")
                continue
        
        return vocab_entries

class AudioTranscriber:
    """Transcripteur audio utilisant WhisperX"""
    
    def __init__(self, language: str = "de"):
        self.model = CONFIG['whisper_model']
        self.language = language
    
    def transcribe(self, audio_path: str, output_dir: str) -> Optional[str]:
        """Transcrit un fichier audio"""
        print(f"Transcription de: {audio_path}")
        
        base_name = Path(audio_path).stem
        output_file = Path(output_dir) / f"{base_name}.json"
        
        command = [
            sys.executable, "-m", "whisperx",
            audio_path,
            "--model", self.model,
            "--language", self.language,
            "--output_format", "json",
            "--output_dir", output_dir,
            "--compute_type", "float32"
        ]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            if output_file.exists():
                print("✓ Transcription terminée")
                return str(output_file)
            else:
                print("✗ Fichier de sortie non trouvé")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Erreur de transcription: {e}")
            return None

class LightProcessor:
    """Processeur principal simplifié"""
    
    def __init__(self, source_lang: str = "de", target_lang: str = "fr"):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.lang_pair = f"{source_lang}-{target_lang}"
        
        # Initialiser les composants
        self.transcriber = AudioTranscriber(source_lang)
        self.translator = SimpleTranslator(self.lang_pair)
        self.vocab_extractor = VocabularyExtractor(
            self.translator, source_lang, target_lang
        )
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Valide le fichier audio"""
        if not Path(audio_path).exists():
            print(f"✗ Fichier non trouvé: {audio_path}")
            return False
        
        if not audio_path.lower().endswith(AUDIO_EXTENSIONS):
            print(f"✗ Format non supporté. Formats acceptés: {', '.join(AUDIO_EXTENSIONS)}")
            return False
        
        return True
    
    def process_transcription(self, json_path: str) -> bool:
        """Traite le fichier de transcription"""
        try:
            # Charger la transcription
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'segments' not in data:
                print("✗ Format JSON invalide")
                return False
            
            # Traduire les segments
            print("Traduction des segments...")
            for segment in tqdm(data['segments'], desc="Segments"):
                text = segment.get('text', '').strip()
                if text:
                    translation = self.translator.translate(text)
                    segment['translation'] = translation
            
            # Extraire le vocabulaire
            print("Extraction du vocabulaire...")
            vocabulary = self.vocab_extractor.extract_vocabulary(data['segments'])
            
            # Ajouter au JSON
            data['vocabulary'] = vocabulary
            data['vocabulary_stats'] = {
                'total_words': len(vocabulary),
                'source_language': self.source_lang,
                'target_language': self.target_lang
            }
            
            # Sauvegarder
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print("✓ Traitement terminé")
            return True
            
        except Exception as e:
            print(f"✗ Erreur: {e}")
            return False
    
    def process_audio(self, audio_path: str) -> bool:
        """Traite un fichier audio complet"""
        # Validation
        if not self.validate_audio_file(audio_path):
            return False
        
        # Créer le dossier de sortie
        output_dir = str(get_output_folder(audio_path))
        print(f"Dossier de sortie: {output_dir}")
        
        # Transcription
        print("\n=== TRANSCRIPTION ===")
        json_path = self.transcriber.transcribe(audio_path, output_dir)
        if not json_path:
            return False
        
        # Traitement
        print("\n=== TRAITEMENT ===")
        success = self.process_transcription(json_path)
        
        if success:
            print(f"\n✓ Traitement complet terminé!")
            print(f"Résultat: {json_path}")
        
        return success

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Version légère du transcripteur audio avec traduction",
        epilog="Exemple: python script_light.py audio.mp3 --source de --target fr"
    )
    
    parser.add_argument("audio_file", help="Fichier audio à traiter")
    parser.add_argument("--source", default="de", help="Langue source (défaut: de)")
    parser.add_argument("--target", default="fr", help="Langue cible (défaut: fr)")
    parser.add_argument("--model", default="base", choices=['tiny', 'base', 'small', 'medium'], 
                       help="Modèle Whisper (défaut: base)")
    parser.add_argument("--max-vocab", type=int, default=50, help="Nombre max de mots (défaut: 50)")
    
    args = parser.parse_args()
    
    # Vérifier la paire de langues
    lang_pair = f"{args.source}-{args.target}"
    if lang_pair not in CONFIG['translation_models']:
        print(f"✗ Paire non supportée: {lang_pair}")
        print(f"Disponibles: {', '.join(CONFIG['translation_models'].keys())}")
        sys.exit(1)
    
    # Configuration
    CONFIG['whisper_model'] = args.model
    CONFIG['max_vocabulary_size'] = args.max_vocab
    
    # Traitement
    processor = LightProcessor(args.source, args.target)
    success = processor.process_audio(args.audio_file)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()