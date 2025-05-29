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
    'max_vocabulary_size': 100,
    'min_word_length': 2,
    'max_examples_per_word': 3,
    'max_exercises_per_type': 4,  # 2 DE->FR + 2 FR->DE
    'min_distractors': 3
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
        
        # Dictionnaire qui sera rempli dynamiquement UNIQUEMENT avec les mots trouvés dans l'audio
        self.discovered_words = {
            'verbs': {'beginner': set(), 'intermediate': set(), 'advanced': set()},
            'nouns': {'beginner': set(), 'intermediate': set(), 'advanced': set()},
            'adjectives': {'beginner': set(), 'intermediate': set(), 'advanced': set()},
            'time_words': {'beginner': set(), 'intermediate': set(), 'advanced': set()},
            'other': {'beginner': set(), 'intermediate': set(), 'advanced': set()}
        }
        
        # Liste complète de tous les mots découverts pour référence
        self.all_discovered_words = []
    
    def detect_word_type(self, word: str) -> str:
        """Détecte le type de mot en analysant sa forme morphologique"""
        word_lower = word.lower()
        
        # Analyse morphologique basée sur les terminaisons allemandes
        # Verbes: mots se terminant en -en, -eln, -ern
        if re.search(r'(en|eln|ern)
    
    def determine_word_level(self, word: str) -> str:
        """Détermine le niveau de difficulté du mot basé sur sa longueur et complexité"""
        # Heuristique basée sur la longueur du mot
        if len(word) <= 4:
            return 'beginner'
        elif len(word) <= 8:
            return 'intermediate'
        else:
            return 'advanced'
    
    def add_to_word_category(self, word: str) -> None:
        """Ajoute un mot à sa catégorie appropriée basée sur les mots découverts"""
        word_type = self.detect_word_type(word)
        level = self.determine_word_level(word)
        
        # Ajouter à la structure organisée par type et niveau
        self.discovered_words[word_type][level].add(word)
        
        # Ajouter à la liste complète
        if word not in self.all_discovered_words:
            self.all_discovered_words.append(word)
    
    def clean_word(self, word: str) -> Optional[str]:
        """Nettoie et valide un mot"""
        if not word:
            return None
            
        # Nettoyer le mot
        word = word.strip()
        word = re.sub(r'^[^\w\säöüß]*|[^\w\säöüß]*$', '', word)
        
        # Valider le mot
        if (len(word) < CONFIG['min_word_length'] or 
            len(word) > 25 or 
            not re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', word) or
            word.lower() in self.processed_words):
            return None
            
        self.processed_words.add(word.lower())
        self.add_to_word_category(word)
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
    
    def get_translation(self, text: str, direction: str = 'de_fr') -> str:
        """Obtient la traduction d'un texte"""
        if direction == 'de_fr':
            tokenizer, model = self.models['de_fr']
        else:
            # Pour FR->DE, on inverse le processus
            tokenizer, model = self.models['de_fr']
        
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=4,
                temperature=0.3,
                do_sample=False,
                early_stopping=True
            )
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return translation if translation else text
            
        except Exception as e:
            print(f"Erreur de traduction pour '{text}': {e}")
            return text
    
    def get_quality_distractors(self, word: str, word_type: str, level: str, count: int = 3) -> List[str]:
        """Génère des distracteurs de qualité pour un mot donné"""
        distractors = []
        
        # 1. Distracteurs de la base de données (même type et niveau)
        if word_type in self.word_database and level in self.word_database[word_type]:
            same_level_words = [w for w in self.word_database[word_type][level] 
                              if w.lower() != word.lower()]
            if same_level_words:
                distractors.extend(random.sample(same_level_words, 
                                               min(2, len(same_level_words))))
        
        # 2. Distracteurs des mots découverts dans le texte
        discovered_same_type = [w for w in self.discovered_words[word_type] 
                               if w.lower() != word.lower()]
        if discovered_same_type and len(distractors) < count:
            remaining_slots = count - len(distractors)
            distractors.extend(random.sample(discovered_same_type, 
                                           min(remaining_slots, len(discovered_same_type))))
        
        # 3. Distracteurs d'autres niveaux du même type si nécessaire
        if len(distractors) < count:
            other_levels = [l for l in ['beginner', 'intermediate', 'advanced'] if l != level]
            for other_level in other_levels:
                if (word_type in self.word_database and 
                    other_level in self.word_database[word_type]):
                    other_words = [w for w in self.word_database[word_type][other_level] 
                                  if w.lower() != word.lower() and w not in distractors]
                    if other_words and len(distractors) < count:
                        remaining_slots = count - len(distractors)
                        distractors.extend(random.sample(other_words, 
                                                       min(remaining_slots, len(other_words))))
        
        # 4. Distracteurs morphologiques (modification du mot original)
        if len(distractors) < count:
            morphological_distractors = self.generate_morphological_distractors(word, word_type)
            for dist in morphological_distractors:
                if len(distractors) < count and dist not in distractors:
                    distractors.append(dist)
        
        return distractors[:count]
    
    def generate_morphological_distractors(self, word: str, word_type: str) -> List[str]:
        """Génère des distracteurs basés sur des modifications morphologiques"""
        distractors = []
        
        if word_type == 'verbs':
            # Pour les verbes, créer des variations de conjugaison
            if word.endswith('en'):
                base = word[:-2]
                distractors.extend([base + 'te', base + 'st', base + 'eln'])
            
        elif word_type == 'nouns':
            # Pour les noms, créer des variations de déclinaison
            distractors.extend([word + 's', word + 'es', word + 'er'])
            
        elif word_type == 'adjectives':
            # Pour les adjectifs, créer des variations
            if not word.endswith('e'):
                distractors.append(word + 'e')
            distractors.extend([word + 'er', word + 'es'])
        
        # Nettoyer et valider les distracteurs
        valid_distractors = []
        for dist in distractors:
            if (len(dist) >= CONFIG['min_word_length'] and 
                len(dist) <= 20 and 
                re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', dist)):
                valid_distractors.append(dist)
        
        return valid_distractors
    
    def create_mcq_exercise(self, word: str, translation: str, word_type: str, 
                          level: str, direction: str = 'de_fr') -> Dict:
        """Crée un exercice QCM dans la direction spécifiée"""
        
        # Obtenir des distracteurs de qualité
        if direction == 'de_fr':
            # Question en français, réponses en allemand
            question_word = translation
            correct_answer = word
            # Obtenir distracteurs allemands
            distractors = self.get_quality_distractors(word, word_type, level)
            # Traduire les distracteurs pour avoir leurs équivalents français
            distractor_translations = [self.get_translation(d, 'de_fr') for d in distractors]
            
        else:  # fr_de
            # Question en allemand, réponses en français
            question_word = word
            correct_answer = translation
            # Obtenir distracteurs allemands puis les traduire
            german_distractors = self.get_quality_distractors(word, word_type, level)
            distractors = [self.get_translation(d, 'de_fr') for d in german_distractors]
        
        # S'assurer qu'on a assez de distracteurs uniques
        unique_distractors = []
        for dist in distractors:
            if (dist != correct_answer and 
                dist.lower() != correct_answer.lower() and
                dist not in unique_distractors):
                unique_distractors.append(dist)
        
        # Compléter avec des distracteurs génériques si nécessaire
        while len(unique_distractors) < CONFIG['min_distractors']:
            if direction == 'de_fr':
                generic_dist = f"Option{len(unique_distractors) + 1}"
            else:
                generic_dist = f"Option{len(unique_distractors) + 1}"
            unique_distractors.append(generic_dist)
        
        # Créer les options de réponse
        options = [correct_answer] + unique_distractors[:CONFIG['min_distractors']]
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        # Templates de questions selon le type et la direction
        if direction == 'de_fr':
            question_templates = {
                'verbs': f"Quel verbe allemand signifie '{question_word}' ?",
                'nouns': f"Quel nom allemand signifie '{question_word}' ?",
                'adjectives': f"Quel adjectif allemand signifie '{question_word}' ?",
                'time_words': f"Quelle expression temporelle allemande signifie '{question_word}' ?",
                'other': f"Quel mot allemand signifie '{question_word}' ?"
            }
        else:  # fr_de
            question_templates = {
                'verbs': f"Que signifie le verbe allemand '{question_word}' ?",
                'nouns': f"Que signifie le nom allemand '{question_word}' ?",
                'adjectives': f"Que signifie l'adjectif allemand '{question_word}' ?",
                'time_words': f"Que signifie l'expression temporelle '{question_word}' ?",
                'other': f"Que signifie le mot allemand '{question_word}' ?"
            }
        
        question_text = question_templates.get(word_type, question_templates['other'])
        
        exercise = {
            'type': 'multiple_choice',
            'direction': direction,
            'difficulty': level,
            'word_type': word_type,
            'question': question_text,
            'options': options,
            'correct_answer': correct_index,
            'correct_option': correct_answer,
            'source_word': word if direction == 'de_fr' else translation,
            'target_word': translation if direction == 'de_fr' else word,
            'explanation': f"'{word}' se traduit par '{translation}'"
        }
        
        return exercise
    
    def generate_example_sentences(self, word: str, count: int = 2) -> List[Dict[str, str]]:
        """Génère des phrases d'exemple pour un mot"""
        examples = []
        
        # Templates prédéfinis par type de mot
        word_type = self.detect_word_type(word)
        
        if word_type == 'verbs':
            templates = [
                f"Ich möchte {word}.",
                f"Wir können {word}.",
                f"Sie will {word}."
            ]
        elif word_type == 'nouns':
            templates = [
                f"Das ist ein {word}.",
                f"Ich habe einen {word}.",
                f"Der {word} ist interessant."
            ]
        elif word_type == 'adjectives':
            templates = [
                f"Das Haus ist {word}.",
                f"Er ist sehr {word}.",
                f"Die Idee ist {word}."
            ]
        else:
            templates = [
                f"Das ist {word}.",
                f"Hier ist {word}.",
                f"{word} ist wichtig."
            ]
        
        # Sélectionner et traduire les exemples
        selected_templates = random.sample(templates, min(count, len(templates)))
        
        for template in selected_templates:
            french_translation = self.get_translation(template, 'de_fr')
            examples.append({
                'de': template,
                'fr': french_translation
            })
        
        return examples
    
    def process_word(self, word: str) -> Dict:
        """Traite un mot complet avec traductions, exemples et exercices"""
        # Informations de base
        word_type = self.detect_word_type(word)
        level = self.determine_word_level(word)
        translation = self.get_translation(word, 'de_fr')
        
        # Exemples
        examples = self.generate_example_sentences(word, 2)
        
        # Exercices bidirectionnels
        exercises = [
            self.create_mcq_exercise(word, translation, word_type, level, 'de_fr'),
            self.create_mcq_exercise(word, translation, word_type, level, 'fr_de')
        ]
        
        return {
            'word': word,
            'translation': translation,
            'word_type': word_type,
            'level': level,
            'examples': examples,
            'exercises': exercises,
            'processed_at': self.get_timestamp()
        }
    
    def get_timestamp(self) -> str:
        """Retourne un timestamp pour le traitement"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
                    translation = self.vocabulary_processor.get_translation(text, 'de_fr')
                    segment['translation'] = translation
            
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
            
            # Statistiques sur les exercices
            total_exercises = sum(len(entry['exercises']) for entry in vocabulary_entries)
            de_fr_exercises = sum(1 for entry in vocabulary_entries 
                                for ex in entry['exercises'] if ex['direction'] == 'de_fr')
            fr_de_exercises = sum(1 for entry in vocabulary_entries 
                                for ex in entry['exercises'] if ex['direction'] == 'fr_de')
            
            # Ajouter le vocabulaire au JSON
            data['vocabulary'] = vocabulary_entries
            data['vocabulary_stats'] = {
                'total_words': len(vocabulary_entries),
                'total_exercises': total_exercises,
                'exercises_de_fr': de_fr_exercises,
                'exercises_fr_de': fr_de_exercises,
                'levels': {
                    'beginner': len([w for w in vocabulary_entries if w['level'] == 'beginner']),
                    'intermediate': len([w for w in vocabulary_entries if w['level'] == 'intermediate']),
                    'advanced': len([w for w in vocabulary_entries if w['level'] == 'advanced'])
                },
                'word_types': {
                    'verbs': len([w for w in vocabulary_entries if w['word_type'] == 'verbs']),
                    'nouns': len([w for w in vocabulary_entries if w['word_type'] == 'nouns']),
                    'adjectives': len([w for w in vocabulary_entries if w['word_type'] == 'adjectives']),
                    'time_words': len([w for w in vocabulary_entries if w['word_type'] == 'time_words']),
                    'other': len([w for w in vocabulary_entries if w['word_type'] == 'other'])
                }
            }
            
            # Sauvegarder le fichier modifié
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print("✓ Traitement du vocabulaire terminé")
            print(f"  → {len(vocabulary_entries)} mots traités")
            print(f"  → {total_exercises} exercices générés ({de_fr_exercises} DE→FR, {fr_de_exercises} FR→DE)")
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
        description="Transcrit et analyse un fichier audio en allemand avec QCM bidirectionnels",
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
    
    parser.add_argument(
        "--min-distractors",
        type=int,
        default=CONFIG['min_distractors'],
        help=f"Nombre minimum de distracteurs par QCM (défaut: {CONFIG['min_distractors']})"
    )
    
    args = parser.parse_args()
    
    # Mettre à jour la configuration
    CONFIG['max_vocabulary_size'] = args.max_vocab
    CONFIG['whisper_model'] = args.model
    CONFIG['min_distractors'] = args.min_distractors
    
    # Traitement
    processor = TranscriptionProcessor()
    success = processor.process_audio_file(args.audio_file, args.output)
    
    sys, word_lower):
            return 'verbs'
            
        # Noms: commencent par une majuscule (en allemand)
        if word[0].isupper():
            return 'nouns'
            
        # Adjectifs: mots se terminant en -ig, -lich, -isch, -bar, -sam
        if re.search(r'(ig|lich|isch|bar|sam)
    
    def determine_word_level(self, word: str) -> str:
        """Détermine le niveau de difficulté du mot"""
        word_type = self.detect_word_type(word)
        
        # Vérifier dans notre base de données
        for level, words in self.word_database[word_type].items():
            if word in words or word.lower() in [w.lower() for w in words]:
                return level
        
        # Heuristique basée sur la longueur si pas dans la base
        if len(word) <= 4:
            return 'beginner'
        elif len(word) <= 8:
            return 'intermediate'
        else:
            return 'advanced'
    
    def add_to_word_category(self, word: str) -> None:
        """Ajoute un mot à sa catégorie appropriée"""
        word_type = self.detect_word_type(word)
        self.discovered_words[word_type].add(word)
    
    def clean_word(self, word: str) -> Optional[str]:
        """Nettoie et valide un mot"""
        if not word:
            return None
            
        # Nettoyer le mot
        word = word.strip()
        word = re.sub(r'^[^\w\säöüß]*|[^\w\säöüß]*$', '', word)
        
        # Valider le mot
        if (len(word) < CONFIG['min_word_length'] or 
            len(word) > 25 or 
            not re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', word) or
            word.lower() in self.processed_words):
            return None
            
        self.processed_words.add(word.lower())
        self.add_to_word_category(word)
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
    
    def get_translation(self, text: str, direction: str = 'de_fr') -> str:
        """Obtient la traduction d'un texte"""
        if direction == 'de_fr':
            tokenizer, model = self.models['de_fr']
        else:
            # Pour FR->DE, on inverse le processus
            tokenizer, model = self.models['de_fr']
        
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=4,
                temperature=0.3,
                do_sample=False,
                early_stopping=True
            )
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return translation if translation else text
            
        except Exception as e:
            print(f"Erreur de traduction pour '{text}': {e}")
            return text
    
    def get_quality_distractors(self, word: str, word_type: str, level: str, count: int = 3) -> List[str]:
        """Génère des distracteurs de qualité pour un mot donné"""
        distractors = []
        
        # 1. Distracteurs de la base de données (même type et niveau)
        if word_type in self.word_database and level in self.word_database[word_type]:
            same_level_words = [w for w in self.word_database[word_type][level] 
                              if w.lower() != word.lower()]
            if same_level_words:
                distractors.extend(random.sample(same_level_words, 
                                               min(2, len(same_level_words))))
        
        # 2. Distracteurs des mots découverts dans le texte
        discovered_same_type = [w for w in self.discovered_words[word_type] 
                               if w.lower() != word.lower()]
        if discovered_same_type and len(distractors) < count:
            remaining_slots = count - len(distractors)
            distractors.extend(random.sample(discovered_same_type, 
                                           min(remaining_slots, len(discovered_same_type))))
        
        # 3. Distracteurs d'autres niveaux du même type si nécessaire
        if len(distractors) < count:
            other_levels = [l for l in ['beginner', 'intermediate', 'advanced'] if l != level]
            for other_level in other_levels:
                if (word_type in self.word_database and 
                    other_level in self.word_database[word_type]):
                    other_words = [w for w in self.word_database[word_type][other_level] 
                                  if w.lower() != word.lower() and w not in distractors]
                    if other_words and len(distractors) < count:
                        remaining_slots = count - len(distractors)
                        distractors.extend(random.sample(other_words, 
                                                       min(remaining_slots, len(other_words))))
        
        # 4. Distracteurs morphologiques (modification du mot original)
        if len(distractors) < count:
            morphological_distractors = self.generate_morphological_distractors(word, word_type)
            for dist in morphological_distractors:
                if len(distractors) < count and dist not in distractors:
                    distractors.append(dist)
        
        return distractors[:count]
    
    def generate_morphological_distractors(self, word: str, word_type: str) -> List[str]:
        """Génère des distracteurs basés sur des modifications morphologiques"""
        distractors = []
        
        if word_type == 'verbs':
            # Pour les verbes, créer des variations de conjugaison
            if word.endswith('en'):
                base = word[:-2]
                distractors.extend([base + 'te', base + 'st', base + 'eln'])
            
        elif word_type == 'nouns':
            # Pour les noms, créer des variations de déclinaison
            distractors.extend([word + 's', word + 'es', word + 'er'])
            
        elif word_type == 'adjectives':
            # Pour les adjectifs, créer des variations
            if not word.endswith('e'):
                distractors.append(word + 'e')
            distractors.extend([word + 'er', word + 'es'])
        
        # Nettoyer et valider les distracteurs
        valid_distractors = []
        for dist in distractors:
            if (len(dist) >= CONFIG['min_word_length'] and 
                len(dist) <= 20 and 
                re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', dist)):
                valid_distractors.append(dist)
        
        return valid_distractors
    
    def create_mcq_exercise(self, word: str, translation: str, word_type: str, 
                          level: str, direction: str = 'de_fr') -> Dict:
        """Crée un exercice QCM dans la direction spécifiée"""
        
        # Obtenir des distracteurs de qualité
        if direction == 'de_fr':
            # Question en français, réponses en allemand
            question_word = translation
            correct_answer = word
            # Obtenir distracteurs allemands
            distractors = self.get_quality_distractors(word, word_type, level)
            # Traduire les distracteurs pour avoir leurs équivalents français
            distractor_translations = [self.get_translation(d, 'de_fr') for d in distractors]
            
        else:  # fr_de
            # Question en allemand, réponses en français
            question_word = word
            correct_answer = translation
            # Obtenir distracteurs allemands puis les traduire
            german_distractors = self.get_quality_distractors(word, word_type, level)
            distractors = [self.get_translation(d, 'de_fr') for d in german_distractors]
        
        # S'assurer qu'on a assez de distracteurs uniques
        unique_distractors = []
        for dist in distractors:
            if (dist != correct_answer and 
                dist.lower() != correct_answer.lower() and
                dist not in unique_distractors):
                unique_distractors.append(dist)
        
        # Compléter avec des distracteurs génériques si nécessaire
        while len(unique_distractors) < CONFIG['min_distractors']:
            if direction == 'de_fr':
                generic_dist = f"Option{len(unique_distractors) + 1}"
            else:
                generic_dist = f"Option{len(unique_distractors) + 1}"
            unique_distractors.append(generic_dist)
        
        # Créer les options de réponse
        options = [correct_answer] + unique_distractors[:CONFIG['min_distractors']]
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        # Templates de questions selon le type et la direction
        if direction == 'de_fr':
            question_templates = {
                'verbs': f"Quel verbe allemand signifie '{question_word}' ?",
                'nouns': f"Quel nom allemand signifie '{question_word}' ?",
                'adjectives': f"Quel adjectif allemand signifie '{question_word}' ?",
                'time_words': f"Quelle expression temporelle allemande signifie '{question_word}' ?",
                'other': f"Quel mot allemand signifie '{question_word}' ?"
            }
        else:  # fr_de
            question_templates = {
                'verbs': f"Que signifie le verbe allemand '{question_word}' ?",
                'nouns': f"Que signifie le nom allemand '{question_word}' ?",
                'adjectives': f"Que signifie l'adjectif allemand '{question_word}' ?",
                'time_words': f"Que signifie l'expression temporelle '{question_word}' ?",
                'other': f"Que signifie le mot allemand '{question_word}' ?"
            }
        
        question_text = question_templates.get(word_type, question_templates['other'])
        
        exercise = {
            'type': 'multiple_choice',
            'direction': direction,
            'difficulty': level,
            'word_type': word_type,
            'question': question_text,
            'options': options,
            'correct_answer': correct_index,
            'correct_option': correct_answer,
            'source_word': word if direction == 'de_fr' else translation,
            'target_word': translation if direction == 'de_fr' else word,
            'explanation': f"'{word}' se traduit par '{translation}'"
        }
        
        return exercise
    
    def generate_example_sentences(self, word: str, count: int = 2) -> List[Dict[str, str]]:
        """Génère des phrases d'exemple pour un mot"""
        examples = []
        
        # Templates prédéfinis par type de mot
        word_type = self.detect_word_type(word)
        
        if word_type == 'verbs':
            templates = [
                f"Ich möchte {word}.",
                f"Wir können {word}.",
                f"Sie will {word}."
            ]
        elif word_type == 'nouns':
            templates = [
                f"Das ist ein {word}.",
                f"Ich habe einen {word}.",
                f"Der {word} ist interessant."
            ]
        elif word_type == 'adjectives':
            templates = [
                f"Das Haus ist {word}.",
                f"Er ist sehr {word}.",
                f"Die Idee ist {word}."
            ]
        else:
            templates = [
                f"Das ist {word}.",
                f"Hier ist {word}.",
                f"{word} ist wichtig."
            ]
        
        # Sélectionner et traduire les exemples
        selected_templates = random.sample(templates, min(count, len(templates)))
        
        for template in selected_templates:
            french_translation = self.get_translation(template, 'de_fr')
            examples.append({
                'de': template,
                'fr': french_translation
            })
        
        return examples
    
    def process_word(self, word: str) -> Dict:
        """Traite un mot complet avec traductions, exemples et exercices"""
        # Informations de base
        word_type = self.detect_word_type(word)
        level = self.determine_word_level(word)
        translation = self.get_translation(word, 'de_fr')
        
        # Exemples
        examples = self.generate_example_sentences(word, 2)
        
        # Exercices bidirectionnels
        exercises = [
            self.create_mcq_exercise(word, translation, word_type, level, 'de_fr'),
            self.create_mcq_exercise(word, translation, word_type, level, 'fr_de')
        ]
        
        return {
            'word': word,
            'translation': translation,
            'word_type': word_type,
            'level': level,
            'examples': examples,
            'exercises': exercises,
            'processed_at': self.get_timestamp()
        }
    
    def get_timestamp(self) -> str:
        """Retourne un timestamp pour le traitement"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
                    translation = self.vocabulary_processor.get_translation(text, 'de_fr')
                    segment['translation'] = translation
            
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
            
            # Statistiques sur les exercices
            total_exercises = sum(len(entry['exercises']) for entry in vocabulary_entries)
            de_fr_exercises = sum(1 for entry in vocabulary_entries 
                                for ex in entry['exercises'] if ex['direction'] == 'de_fr')
            fr_de_exercises = sum(1 for entry in vocabulary_entries 
                                for ex in entry['exercises'] if ex['direction'] == 'fr_de')
            
            # Ajouter le vocabulaire au JSON
            data['vocabulary'] = vocabulary_entries
            data['vocabulary_stats'] = {
                'total_words': len(vocabulary_entries),
                'total_exercises': total_exercises,
                'exercises_de_fr': de_fr_exercises,
                'exercises_fr_de': fr_de_exercises,
                'levels': {
                    'beginner': len([w for w in vocabulary_entries if w['level'] == 'beginner']),
                    'intermediate': len([w for w in vocabulary_entries if w['level'] == 'intermediate']),
                    'advanced': len([w for w in vocabulary_entries if w['level'] == 'advanced'])
                },
                'word_types': {
                    'verbs': len([w for w in vocabulary_entries if w['word_type'] == 'verbs']),
                    'nouns': len([w for w in vocabulary_entries if w['word_type'] == 'nouns']),
                    'adjectives': len([w for w in vocabulary_entries if w['word_type'] == 'adjectives']),
                    'time_words': len([w for w in vocabulary_entries if w['word_type'] == 'time_words']),
                    'other': len([w for w in vocabulary_entries if w['word_type'] == 'other'])
                }
            }
            
            # Sauvegarder le fichier modifié
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print("✓ Traitement du vocabulaire terminé")
            print(f"  → {len(vocabulary_entries)} mots traités")
            print(f"  → {total_exercises} exercices générés ({de_fr_exercises} DE→FR, {fr_de_exercises} FR→DE)")
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
        description="Transcrit et analyse un fichier audio en allemand avec QCM bidirectionnels",
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
    
    parser.add_argument(
        "--min-distractors",
        type=int,
        default=CONFIG['min_distractors'],
        help=f"Nombre minimum de distracteurs par QCM (défaut: {CONFIG['min_distractors']})"
    )
    
    args = parser.parse_args()
    
    # Mettre à jour la configuration
    CONFIG['max_vocabulary_size'] = args.max_vocab
    CONFIG['whisper_model'] = args.model
    CONFIG['min_distractors'] = args.min_distractors
    
    # Traitement
    processor = TranscriptionProcessor()
    success = processor.process_audio_file(args.audio_file, args.output)
    
    sys, word_lower):
            return 'adjectives'
            
        # Mots de temps: patterns spécifiques communs
        time_patterns = ['zeit', 'tag', 'jahr', 'monat', 'woche', 'stunde', 'minute']
        if any(pattern in word_lower for pattern in time_patterns):
            return 'time_words'
        
        # Mots de temps spécifiques courants
        common_time_words = ['heute', 'morgen', 'gestern', 'jetzt', 'dann', 'immer', 'nie', 'oft', 'selten', 'später', 'früh', 'spät']
        if word_lower in common_time_words:
            return 'time_words'
            
        return 'other'
    
    def determine_word_level(self, word: str) -> str:
        """Détermine le niveau de difficulté du mot"""
        word_type = self.detect_word_type(word)
        
        # Vérifier dans notre base de données
        for level, words in self.word_database[word_type].items():
            if word in words or word.lower() in [w.lower() for w in words]:
                return level
        
        # Heuristique basée sur la longueur si pas dans la base
        if len(word) <= 4:
            return 'beginner'
        elif len(word) <= 8:
            return 'intermediate'
        else:
            return 'advanced'
    
    def add_to_word_category(self, word: str) -> None:
        """Ajoute un mot à sa catégorie appropriée"""
        word_type = self.detect_word_type(word)
        self.discovered_words[word_type].add(word)
    
    def clean_word(self, word: str) -> Optional[str]:
        """Nettoie et valide un mot"""
        if not word:
            return None
            
        # Nettoyer le mot
        word = word.strip()
        word = re.sub(r'^[^\w\säöüß]*|[^\w\säöüß]*$', '', word)
        
        # Valider le mot
        if (len(word) < CONFIG['min_word_length'] or 
            len(word) > 25 or 
            not re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', word) or
            word.lower() in self.processed_words):
            return None
            
        self.processed_words.add(word.lower())
        self.add_to_word_category(word)
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
    
    def get_translation(self, text: str, direction: str = 'de_fr') -> str:
        """Obtient la traduction d'un texte"""
        if direction == 'de_fr':
            tokenizer, model = self.models['de_fr']
        else:
            # Pour FR->DE, on inverse le processus
            tokenizer, model = self.models['de_fr']
        
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=4,
                temperature=0.3,
                do_sample=False,
                early_stopping=True
            )
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return translation if translation else text
            
        except Exception as e:
            print(f"Erreur de traduction pour '{text}': {e}")
            return text
    
    def get_quality_distractors(self, word: str, word_type: str, level: str, count: int = 3) -> List[str]:
        """Génère des distracteurs de qualité pour un mot donné"""
        distractors = []
        
        # 1. Distracteurs de la base de données (même type et niveau)
        if word_type in self.word_database and level in self.word_database[word_type]:
            same_level_words = [w for w in self.word_database[word_type][level] 
                              if w.lower() != word.lower()]
            if same_level_words:
                distractors.extend(random.sample(same_level_words, 
                                               min(2, len(same_level_words))))
        
        # 2. Distracteurs des mots découverts dans le texte
        discovered_same_type = [w for w in self.discovered_words[word_type] 
                               if w.lower() != word.lower()]
        if discovered_same_type and len(distractors) < count:
            remaining_slots = count - len(distractors)
            distractors.extend(random.sample(discovered_same_type, 
                                           min(remaining_slots, len(discovered_same_type))))
        
        # 3. Distracteurs d'autres niveaux du même type si nécessaire
        if len(distractors) < count:
            other_levels = [l for l in ['beginner', 'intermediate', 'advanced'] if l != level]
            for other_level in other_levels:
                if (word_type in self.word_database and 
                    other_level in self.word_database[word_type]):
                    other_words = [w for w in self.word_database[word_type][other_level] 
                                  if w.lower() != word.lower() and w not in distractors]
                    if other_words and len(distractors) < count:
                        remaining_slots = count - len(distractors)
                        distractors.extend(random.sample(other_words, 
                                                       min(remaining_slots, len(other_words))))
        
        # 4. Distracteurs morphologiques (modification du mot original)
        if len(distractors) < count:
            morphological_distractors = self.generate_morphological_distractors(word, word_type)
            for dist in morphological_distractors:
                if len(distractors) < count and dist not in distractors:
                    distractors.append(dist)
        
        return distractors[:count]
    
    def generate_morphological_distractors(self, word: str, word_type: str) -> List[str]:
        """Génère des distracteurs basés sur des modifications morphologiques"""
        distractors = []
        
        if word_type == 'verbs':
            # Pour les verbes, créer des variations de conjugaison
            if word.endswith('en'):
                base = word[:-2]
                distractors.extend([base + 'te', base + 'st', base + 'eln'])
            
        elif word_type == 'nouns':
            # Pour les noms, créer des variations de déclinaison
            distractors.extend([word + 's', word + 'es', word + 'er'])
            
        elif word_type == 'adjectives':
            # Pour les adjectifs, créer des variations
            if not word.endswith('e'):
                distractors.append(word + 'e')
            distractors.extend([word + 'er', word + 'es'])
        
        # Nettoyer et valider les distracteurs
        valid_distractors = []
        for dist in distractors:
            if (len(dist) >= CONFIG['min_word_length'] and 
                len(dist) <= 20 and 
                re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', dist)):
                valid_distractors.append(dist)
        
        return valid_distractors
    
    def create_mcq_exercise(self, word: str, translation: str, word_type: str, 
                          level: str, direction: str = 'de_fr') -> Dict:
        """Crée un exercice QCM dans la direction spécifiée"""
        
        # Obtenir des distracteurs de qualité
        if direction == 'de_fr':
            # Question en français, réponses en allemand
            question_word = translation
            correct_answer = word
            # Obtenir distracteurs allemands
            distractors = self.get_quality_distractors(word, word_type, level)
            # Traduire les distracteurs pour avoir leurs équivalents français
            distractor_translations = [self.get_translation(d, 'de_fr') for d in distractors]
            
        else:  # fr_de
            # Question en allemand, réponses en français
            question_word = word
            correct_answer = translation
            # Obtenir distracteurs allemands puis les traduire
            german_distractors = self.get_quality_distractors(word, word_type, level)
            distractors = [self.get_translation(d, 'de_fr') for d in german_distractors]
        
        # S'assurer qu'on a assez de distracteurs uniques
        unique_distractors = []
        for dist in distractors:
            if (dist != correct_answer and 
                dist.lower() != correct_answer.lower() and
                dist not in unique_distractors):
                unique_distractors.append(dist)
        
        # Compléter avec des distracteurs génériques si nécessaire
        while len(unique_distractors) < CONFIG['min_distractors']:
            if direction == 'de_fr':
                generic_dist = f"Option{len(unique_distractors) + 1}"
            else:
                generic_dist = f"Option{len(unique_distractors) + 1}"
            unique_distractors.append(generic_dist)
        
        # Créer les options de réponse
        options = [correct_answer] + unique_distractors[:CONFIG['min_distractors']]
        random.shuffle(options)
        correct_index = options.index(correct_answer)
        
        # Templates de questions selon le type et la direction
        if direction == 'de_fr':
            question_templates = {
                'verbs': f"Quel verbe allemand signifie '{question_word}' ?",
                'nouns': f"Quel nom allemand signifie '{question_word}' ?",
                'adjectives': f"Quel adjectif allemand signifie '{question_word}' ?",
                'time_words': f"Quelle expression temporelle allemande signifie '{question_word}' ?",
                'other': f"Quel mot allemand signifie '{question_word}' ?"
            }
        else:  # fr_de
            question_templates = {
                'verbs': f"Que signifie le verbe allemand '{question_word}' ?",
                'nouns': f"Que signifie le nom allemand '{question_word}' ?",
                'adjectives': f"Que signifie l'adjectif allemand '{question_word}' ?",
                'time_words': f"Que signifie l'expression temporelle '{question_word}' ?",
                'other': f"Que signifie le mot allemand '{question_word}' ?"
            }
        
        question_text = question_templates.get(word_type, question_templates['other'])
        
        exercise = {
            'type': 'multiple_choice',
            'direction': direction,
            'difficulty': level,
            'word_type': word_type,
            'question': question_text,
            'options': options,
            'correct_answer': correct_index,
            'correct_option': correct_answer,
            'source_word': word if direction == 'de_fr' else translation,
            'target_word': translation if direction == 'de_fr' else word,
            'explanation': f"'{word}' se traduit par '{translation}'"
        }
        
        return exercise
    
    def generate_example_sentences(self, word: str, count: int = 2) -> List[Dict[str, str]]:
        """Génère des phrases d'exemple pour un mot"""
        examples = []
        
        # Templates prédéfinis par type de mot
        word_type = self.detect_word_type(word)
        
        if word_type == 'verbs':
            templates = [
                f"Ich möchte {word}.",
                f"Wir können {word}.",
                f"Sie will {word}."
            ]
        elif word_type == 'nouns':
            templates = [
                f"Das ist ein {word}.",
                f"Ich habe einen {word}.",
                f"Der {word} ist interessant."
            ]
        elif word_type == 'adjectives':
            templates = [
                f"Das Haus ist {word}.",
                f"Er ist sehr {word}.",
                f"Die Idee ist {word}."
            ]
        else:
            templates = [
                f"Das ist {word}.",
                f"Hier ist {word}.",
                f"{word} ist wichtig."
            ]
        
        # Sélectionner et traduire les exemples
        selected_templates = random.sample(templates, min(count, len(templates)))
        
        for template in selected_templates:
            french_translation = self.get_translation(template, 'de_fr')
            examples.append({
                'de': template,
                'fr': french_translation
            })
        
        return examples
    
    def process_word(self, word: str) -> Dict:
        """Traite un mot complet avec traductions, exemples et exercices"""
        # Informations de base
        word_type = self.detect_word_type(word)
        level = self.determine_word_level(word)
        translation = self.get_translation(word, 'de_fr')
        
        # Exemples
        examples = self.generate_example_sentences(word, 2)
        
        # Exercices bidirectionnels
        exercises = [
            self.create_mcq_exercise(word, translation, word_type, level, 'de_fr'),
            self.create_mcq_exercise(word, translation, word_type, level, 'fr_de')
        ]
        
        return {
            'word': word,
            'translation': translation,
            'word_type': word_type,
            'level': level,
            'examples': examples,
            'exercises': exercises,
            'processed_at': self.get_timestamp()
        }
    
    def get_timestamp(self) -> str:
        """Retourne un timestamp pour le traitement"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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