#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    GPT2LMHeadModel, 
    AutoTokenizer
)
from tqdm import tqdm

# Configuration globale
CONFIG = {
    'whisper_model': "base",
    'language': "de", 
    'output_format': "json",
    'translation_model_path': "./opus-mt-de-fr",
    'german_gpt_model': "benjamin/gpt2-wechsel-german",
    'max_vocabulary_size': 100,  # Limiter le nombre de mots à traiter
    'min_word_length': 2,
    'max_examples_per_word': 3,
    'max_exercises_per_type': 2
}

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

class VocabularyProcessor:
    """Classe pour gérer le traitement du vocabulaire et les analyses lexicales"""
    
    def __init__(self, models: Dict):
        self.models = models
        self.processed_words = set()
        
    def clean_word(self, word: str) -> Optional[str]:
        """Nettoie et valide un mot"""
        if not word:
            return None
            
        # Nettoyer le mot
        word = word.strip().lower()
        word = re.sub(r'^[^\w\säöüß]*|[^\w\säöüß]*$', '', word)
        
        # Valider le mot
        if (len(word) < CONFIG['min_word_length'] or 
            len(word) > 20 or 
            not re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', word) or
            word in self.processed_words):
            return None
            
        self.processed_words.add(word)
        return word
    
    def extract_vocabulary(self, segments: List[Dict]) -> List[str]:
        """Extrait et nettoie le vocabulaire des segments transcrits"""
        vocabulary = set()
        
        for segment in segments:
            text = segment.get('text', '').strip()
            words = text.split()
            
            for word in words:
                clean_word = self.clean_word(word)
                if clean_word and len(vocabulary) < CONFIG['max_vocabulary_size']:
                    vocabulary.add(clean_word)
        
        return sorted(list(vocabulary))
    
    def get_basic_translation(self, word: str) -> Dict[str, str]:
        """Obtient la traduction de base d'un mot"""
        tokenizer_de_fr, model_de_fr = self.models['de_fr']
        
        try:
            inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=20,
                num_beams=3,
                temperature=0.3,
                do_sample=False
            )
            translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True).strip()
            
            return {
                'de': word,
                'fr': translation if translation else word
            }
        except Exception as e:
            print(f"Erreur de traduction pour '{word}': {e}")
            return {'de': word, 'fr': word}
    
    def generate_simple_example(self, word: str) -> Dict[str, str]:
        """Génère un exemple simple d'utilisation"""
        tokenizer_gpt, model_gpt = self.models['gpt']
        tokenizer_de_fr, model_de_fr = self.models['de_fr']
        
        # Templates simples prédéfinis
        simple_templates = [
            f"Das ist ein {word}.",
            f"Ich habe einen {word}.",
            f"Der {word} ist schön.",
            f"Wir brauchen {word}."
        ]
        
        try:
            # Choisir un template au hasard
            german_sentence = random.choice(simple_templates)
            
            # Traduire en français
            inputs = tokenizer_de_fr(german_sentence, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=30,
                num_beams=3,
                temperature=0.3
            )
            french_sentence = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True).strip()
            
            return {
                'de': german_sentence,
                'fr': french_sentence if french_sentence else "Phrase d'exemple"
            }
            
        except Exception as e:
            print(f"Erreur génération exemple pour '{word}': {e}")
            return {
                'de': f"Das ist {word}.",
                'fr': f"C'est {word}."
            }
    
    def create_simple_exercise(self, word: str, translation: str) -> Dict:
        """Crée un exercice simple à trous"""
        german_sentence = f"Ich brauche einen ____."
        french_instruction = f"Complétez avec le mot allemand pour '{translation}'"
        
        return {
            'type': 'fill_blank',
            'question': {
                'de': german_sentence,
                'fr': french_instruction
            },
            'answer': word,
            'level': 'beginner'
        }
    
    def determine_word_level(self, word: str) -> str:
        """Détermine le niveau de difficulté du mot"""
        if len(word) <= 4:
            return 'beginner'
        elif len(word) <= 8:
            return 'intermediate'
        else:
            return 'advanced'
    
    def process_word(self, word: str) -> Dict:
        """Traite un mot complet avec traduction, exemple et exercice"""
        translation_info = self.get_basic_translation(word)
        example = self.generate_simple_example(word)
        exercise = self.create_simple_exercise(word, translation_info['fr'])
        level = self.determine_word_level(word)
        
        return {
            'word': word,
            'level': level,
            'translation': translation_info,
            'example': example,
            'exercise': exercise,
            'processed_at': self.get_timestamp()
        }
    
    def get_timestamp(self) -> str:
        """Retourne un timestamp pour le traitement"""
        from datetime import datetime
        return datetime.now().isoformat()

class AudioTranscriber:
    """Classe pour gérer la transcription audio"""
    
    def __init__(self):
        self.model = CONFIG['whisper_model']
        self.language = CONFIG['language']
        self.output_format = CONFIG['output_format']
    
    def transcribe_file(self, audio_path: str, output_directory: str) -> Optional[str]:
        """Transcrit un fichier audio"""
        print(f"Transcription de: {audio_path}")
        
        base_name = Path(audio_path).stem
        output_filename = f"{base_name}.{self.output_format}"
        output_file_path = Path(output_directory) / output_filename
        
        command = [
            sys.executable, "-m", "whisperx",
            audio_path,
            "--model", self.model,
            "--language", self.language,
            "--output_format", self.output_format,
            "--output_dir", output_directory,
            "--segment_resolution", "chunk",
            "--max_line_count", "1",
            "--compute_type", "float32"
        ]
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if output_file_path.exists():
                print("✓ Transcription terminée avec succès")
                return str(output_file_path)
            else:
                print("✗ Fichier de sortie non trouvé")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Erreur de transcription: {e}")
            print(f"Sortie d'erreur: {e.stderr}")
            return None
        except Exception as e:
            print(f"✗ Erreur inattendue: {e}")
            return None

class ModelManager:
    """Classe pour gérer le chargement et la gestion des modèles"""
    
    def __init__(self):
        self.models = {}
    
    def load_models(self) -> Dict:
        """Charge tous les modèles nécessaires"""
        print("Chargement des modèles...")
        
        try:
            # Modèle de traduction DE -> FR
            print("→ Chargement du modèle DE->FR...")
            tokenizer_de_fr = MarianTokenizer.from_pretrained(CONFIG['translation_model_path'])
            model_de_fr = MarianMTModel.from_pretrained(CONFIG['translation_model_path'])
            
            # Modèle GPT allemand
            print("→ Chargement du modèle GPT allemand...")
            tokenizer_gpt = AutoTokenizer.from_pretrained(CONFIG['german_gpt_model'])
            model_gpt = GPT2LMHeadModel.from_pretrained(CONFIG['german_gpt_model'])
            
            # Configuration du tokenizer GPT
            if tokenizer_gpt.pad_token is None:
                tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
            
            self.models = {
                'de_fr': (tokenizer_de_fr, model_de_fr),
                'gpt': (tokenizer_gpt, model_gpt)
            }
            
            print("✓ Tous les modèles chargés avec succès")
            return self.models
            
        except Exception as e:
            print(f"✗ Erreur lors du chargement des modèles: {e}")
            raise

class TranscriptionProcessor:
    """Classe principale pour traiter la transcription complète"""
    
    def __init__(self):
        self.transcriber = AudioTranscriber()
        self.model_manager = ModelManager()
        self.vocabulary_processor = None
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Valide le fichier audio"""
        if not Path(audio_path).exists():
            print(f"✗ Fichier non trouvé: {audio_path}")
            return False
        
        if not audio_path.lower().endswith(AUDIO_EXTENSIONS):
            print(f"✗ Format audio non supporté. Formats acceptés: {', '.join(AUDIO_EXTENSIONS)}")
            return False
        
        return True
    
    def process_transcription_file(self, json_path: str) -> bool:
        """Traite le fichier JSON de transcription"""
        try:
            # Charger le fichier JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'segments' not in data:
                print("✗ Format JSON invalide - 'segments' manquant")
                return False
            
            # Initialiser le processeur de vocabulaire
            models = self.model_manager.load_models()
            self.vocabulary_processor = VocabularyProcessor(models)
            
            # Traiter chaque segment et ajouter la traduction
            print("Traduction des segments...")
            for i, segment in enumerate(tqdm(data['segments'], desc="Segments")):
                text = segment.get('text', '').strip()
                if text:
                    translation_info = self.vocabulary_processor.get_basic_translation(text)
                    segment['translation'] = translation_info['fr']
            
            # Extraire et traiter le vocabulaire
            print("Extraction du vocabulaire...")
            vocabulary_words = self.vocabulary_processor.extract_vocabulary(data['segments'])
            
            print(f"Traitement de {len(vocabulary_words)} mots du vocabulaire...")
            vocabulary_entries = []
            
            for word in tqdm(vocabulary_words, desc="Vocabulaire"):
                try:
                    entry = self.vocabulary_processor.process_word(word)
                    vocabulary_entries.append(entry)
                except Exception as e:
                    print(f"Erreur pour le mot '{word}': {e}")
                    continue
            
            # Ajouter le vocabulaire au JSON
            data['vocabulary'] = vocabulary_entries
            data['vocabulary_stats'] = {
                'total_words': len(vocabulary_entries),
                'levels': {
                    'beginner': len([w for w in vocabulary_entries if w['level'] == 'beginner']),
                    'intermediate': len([w for w in vocabulary_entries if w['level'] == 'intermediate']),
                    'advanced': len([w for w in vocabulary_entries if w['level'] == 'advanced'])
                }
            }
            
            # Sauvegarder le fichier modifié
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print("✓ Traitement du vocabulaire terminé")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors du traitement: {e}")
            return False
    
    def process_audio_file(self, audio_path: str, output_dir: str) -> bool:
        """Traite un fichier audio complet"""
        # Validation
        if not self.validate_audio_file(audio_path):
            return False
        
        # Créer le répertoire de sortie
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Étape 1: Transcription
        print("\n=== ÉTAPE 1: TRANSCRIPTION ===")
        json_path = self.transcriber.transcribe_file(audio_path, output_dir)
        if not json_path:
            return False
        
        # Étape 2: Traitement du vocabulaire
        print("\n=== ÉTAPE 2: TRAITEMENT DU VOCABULAIRE ===")
        if not self.process_transcription_file(json_path):
            return False
        
        print(f"\n✓ Traitement complet terminé!")
        print(f"Fichier de sortie: {json_path}")
        return True

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Transcrit et analyse un fichier audio en allemand",
        epilog="Exemple: python script.py audio.mp3 -o ./output"
    )
    
    parser.add_argument(
        "audio_file",
        help="Fichier audio à traiter"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Répertoire de sortie (obligatoire)",
        required=True
    )
    
    parser.add_argument(
        "--max-vocab",
        type=int,
        default=CONFIG['max_vocabulary_size'],
        help=f"Nombre maximum de mots à traiter (défaut: {CONFIG['max_vocabulary_size']})"
    )
    
    parser.add_argument(
        "--model",
        default=CONFIG['whisper_model'],
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help=f"Modèle Whisper à utiliser (défaut: {CONFIG['whisper_model']})"
    )
    
    args = parser.parse_args()
    
    # Mettre à jour la configuration
    CONFIG['max_vocabulary_size'] = args.max_vocab
    CONFIG['whisper_model'] = args.model
    
    # Traitement
    processor = TranscriptionProcessor()
    success = processor.process_audio_file(args.audio_file, args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()