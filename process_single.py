#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import re
import json
import random
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    GPT2LMHeadModel, 
    AutoTokenizer
)
from tqdm import tqdm
import time

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
    'max_exercises_per_type': 2,
    'pronunciation_delay': 3.0,  # Délai entre les requêtes de prononciation
    'max_pronunciation_attempts': 3  # Nombre maximum de tentatives par mot
}

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

class PronunciationScraper:
    """Classe pour scraper les prononciations depuis HowToPronounce.com"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.pronunciation_dir = self.output_dir / "pronunciations"
        self.pronunciation_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8,de;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def get_pronunciation_url(self, word: str, language: str = "german") -> str:
        """Construit l'URL pour HowToPronounce.com"""
        encoded_word = quote(word, safe='')
        return f"https://fr.howtopronounce.com/{language}/{encoded_word}"
    
    def scrape_audio_sources(self, url: str) -> List[Dict]:
        """Scrape les sources audio depuis une page HowToPronounce"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            audio_sources = []
            
            # Méthode 1: Balises audio directes
            audio_tags = soup.find_all('audio')
            for audio in audio_tags:
                if audio.get('src'):
                    src = audio.get('src')
                    full_url = urljoin(url, src)
                    audio_sources.append({
                        'type': 'direct_src',
                        'url': full_url,
                        'original_src': src
                    })
                
                # Balises source enfants
                sources = audio.find_all('source')
                for source in sources:
                    if source.get('src'):
                        src = source.get('src')
                        full_url = urljoin(url, src)
                        audio_sources.append({
                            'type': 'source_tag',
                            'url': full_url,
                            'original_src': src,
                            'mime_type': source.get('type', 'audio/mpeg')
                        })
            
            # Méthode 2: JavaScript embarqué
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    audio_urls = re.findall(
                        r'["\']([^"\']*\.(?:mp3|wav|ogg|m4a)[^"\']*)["\']', 
                        script.string, re.IGNORECASE
                    )
                    for audio_url in audio_urls:
                        full_url = urljoin(url, audio_url)
                        audio_sources.append({
                            'type': 'js_embedded',
                            'url': full_url,
                            'original_src': audio_url
                        })
            
            # Méthode 3: Attributs data-*
            data_elements = soup.find_all(attrs=lambda x: x and any(
                attr.startswith('data-') and isinstance(val, str) and 
                any(ext in val.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a'])
                for attr, val in x.items()
            ))
            
            for element in data_elements:
                for attr, value in element.attrs.items():
                    if (attr.startswith('data-') and isinstance(value, str) and
                        any(ext in value.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a'])):
                        full_url = urljoin(url, value)
                        audio_sources.append({
                            'type': f'data_attribute_{attr}',
                            'url': full_url,
                            'original_src': value
                        })
            
            # Supprimer les doublons
            unique_sources = {}
            for source in audio_sources:
                url_key = source['url']
                if url_key not in unique_sources:
                    unique_sources[url_key] = source
            
            return list(unique_sources.values())
            
        except Exception as e:
            print(f"Erreur lors du scraping de {url}: {e}")
            return []
    
    def download_audio(self, audio_url: str, filename: str) -> bool:
        """Télécharge un fichier audio"""
        try:
            response = requests.get(audio_url, headers=self.headers, stream=True, timeout=15)
            response.raise_for_status()
            
            file_path = self.pronunciation_dir / filename
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Vérifier que le fichier n'est pas vide
            if file_path.stat().st_size > 0:
                return True
            else:
                file_path.unlink()  # Supprimer le fichier vide
                return False
                
        except Exception as e:
            print(f"Erreur lors du téléchargement de {audio_url}: {e}")
            return False
    
    def get_pronunciation_audio(self, word: str) -> Optional[Dict]:
        """Récupère l'audio de prononciation pour un mot"""
        print(f"  → Recherche prononciation pour: {word}")
        
        # Vérifier si le fichier existe déjà
        potential_files = [
            f"{word}_pronunciation.mp3",
            f"{word}_pronunciation.wav",
            f"{word}_pronunciation.ogg"
        ]
        
        for filename in potential_files:
            if (self.pronunciation_dir / filename).exists():
                print(f"  ✓ Prononciation déjà téléchargée: {filename}")
                return {
                    'word': word,
                    'audio_file': str(self.pronunciation_dir / filename),
                    'status': 'cached'
                }
        
        # Essayer de télécharger
        url = self.get_pronunciation_url(word)
        audio_sources = self.scrape_audio_sources(url)
        
        if not audio_sources:
            print(f"  ✗ Aucune source audio trouvée pour: {word}")
            return None
        
        # Essayer de télécharger la première source valide
        for i, source in enumerate(audio_sources[:CONFIG['max_pronunciation_attempts']]):
            filename = f"{word}_pronunciation_{i}.mp3"
            
            if self.download_audio(source['url'], filename):
                print(f"  ✓ Prononciation téléchargée: {filename}")
                return {
                    'word': word,
                    'audio_file': str(self.pronunciation_dir / filename),
                    'source_url': source['url'],
                    'source_type': source['type'],
                    'status': 'downloaded'
                }
            
            time.sleep(0.5)  # Petit délai entre les tentatives
        
        print(f"  ✗ Échec du téléchargement pour: {word}")
        return None
    
    def batch_download_pronunciations(self, words: List[str]) -> Dict[str, Optional[Dict]]:
        """Télécharge les prononciations pour une liste de mots"""
        print(f"\n=== TÉLÉCHARGEMENT DES PRONONCIATIONS ===")
        print(f"Traitement de {len(words)} mots...")
        
        results = {}
        successful_downloads = 0
        
        for i, word in enumerate(tqdm(words, desc="Prononciations")):
            try:
                result = self.get_pronunciation_audio(word)
                results[word] = result
                
                if result and result['status'] in ['downloaded', 'cached']:
                    successful_downloads += 1
                
                # Délai entre les requêtes pour éviter d'être bloqué
                if i < len(words) - 1:  # Pas de délai pour le dernier mot
                    time.sleep(CONFIG['pronunciation_delay'])
                    
            except Exception as e:
                print(f"  ✗ Erreur pour {word}: {e}")
                results[word] = None
        
        print(f"\n✓ Téléchargements terminés: {successful_downloads}/{len(words)} réussis")
        return results

class VocabularyProcessor:
    """Classe pour gérer le traitement du vocabulaire et les analyses lexicales"""
    
    def __init__(self, models: Dict, pronunciation_scraper: PronunciationScraper):
        self.models = models
        self.processed_words = set()
        self.pronunciation_scraper = pronunciation_scraper
        
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
        """Crée deux exercices à choix multiples (QCM) dans les deux sens de traduction"""
        tokenizer_gpt, model_gpt = self.models['gpt']
        
        try:
            # Générer des mots distracteurs
            wrong_options = []
            prompts = [
                "Ein deutsches Wort:",
                "Noch ein Wort:",
                "Weiteres Wort:"
            ]
            
            for prompt in prompts:
                try:
                    inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
                    outputs = model_gpt.generate(
                        inputs.input_ids,
                        max_length=15,
                        num_return_sequences=1,
                        temperature=1.0,
                        do_sample=True,
                        top_k=50
                    )
                    
                    generated_text = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
                    words = re.findall(r'\b[a-zäöüß]{3,}\b', generated_text.lower())
                    
                    if words:
                        candidate = words[0]
                        if candidate != word and candidate not in wrong_options:
                            wrong_options.append(candidate)
                    
                except Exception:
                    continue
            
            # Fallback si pas assez d'options
            while len(wrong_options) < 3:
                fallbacks = ['haus', 'auto', 'buch', 'wort', 'zeit', 'jahr', 'mann', 'frau']
                for fallback in fallbacks:
                    if fallback != word and fallback not in wrong_options:
                        wrong_options.append(fallback)
                        if len(wrong_options) >= 3:
                            break
            
            wrong_options = wrong_options[:3]
            
            # Exercice DE -> FR
            de_fr_options = [translation] + [f"{opt}fr" for opt in wrong_options]
            random.shuffle(de_fr_options)
            
            # Exercice FR -> DE
            fr_de_options = [word] + wrong_options
            random.shuffle(fr_de_options)
            
            return {
                'type': 'multiple_choice_pair',
                'de_to_fr': {
                    'question': f"Quelle est la traduction française de '{word}'?",
                    'options': de_fr_options,
                    'answer': translation
                },
                'fr_to_de': {
                    'question': f"Quel est le mot allemand pour '{translation}'?",
                    'options': fr_de_options,
                    'answer': word
                },
                'level': self.determine_word_level(word)
            }
            
        except Exception as e:
            print(f"Erreur génération exercice pour '{word}': {e}")
            return {
                'type': 'simple',
                'question': f"Traduisez: {word}",
                'answer': translation,
                'level': self.determine_word_level(word)
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
        """Traite un mot complet avec traduction, exemple, exercice et prononciation"""
        translation_info = self.get_basic_translation(word)
        example = self.generate_simple_example(word)
        exercise = self.create_simple_exercise(word, translation_info['fr'])
        level = self.determine_word_level(word)
        
        # Récupérer la prononciation
        pronunciation = self.pronunciation_scraper.get_pronunciation_audio(word)
        
        return {
            'word': word,
            'level': level,
            'translation': translation_info,
            'example': example,
            'exercise': exercise,
            'pronunciation': pronunciation,
            'processed_at': self.get_timestamp()
        }
    
    def process_vocabulary_batch(self, words: List[str]) -> List[Dict]:
        """Traite une liste de mots avec téléchargement groupé des prononciations"""
        print("Traitement du vocabulaire avec prononciations...")
        
        # Télécharger toutes les prononciations en lot
        pronunciation_results = self.pronunciation_scraper.batch_download_pronunciations(words)
        
        # Traiter chaque mot
        vocabulary_entries = []
        for word in tqdm(words, desc="Traitement mots"):
            try:
                translation_info = self.get_basic_translation(word)
                example = self.generate_simple_example(word)
                exercise = self.create_simple_exercise(word, translation_info['fr'])
                level = self.determine_word_level(word)
                
                entry = {
                    'word': word,
                    'level': level,
                    'translation': translation_info,
                    'example': example,
                    'exercise': exercise,
                    'pronunciation': pronunciation_results.get(word),
                    'processed_at': self.get_timestamp()
                }
                
                vocabulary_entries.append(entry)
                
            except Exception as e:
                print(f"Erreur pour le mot '{word}': {e}")
                continue
        
        return vocabulary_entries
    
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
        self.pronunciation_scraper = None
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Valide le fichier audio"""
        if not Path(audio_path).exists():
            print(f"✗ Fichier non trouvé: {audio_path}")
            return False
        
        if not audio_path.lower().endswith(AUDIO_EXTENSIONS):
            print(f"✗ Format audio non supporté. Formats acceptés: {', '.join(AUDIO_EXTENSIONS)}")
            return False
        
        return True
    
    def process_transcription_file(self, json_path: str, output_dir: str) -> bool:
        """Traite le fichier JSON de transcription"""
        try:
            # Charger le fichier JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'segments' not in data:
                print("✗ Format JSON invalide - 'segments' manquant")
                return False
            
            # Initialiser les processeurs
            models = self.model_manager.load_models()
            self.pronunciation_scraper = PronunciationScraper(output_dir)
            self.vocabulary_processor = VocabularyProcessor(models, self.pronunciation_scraper)
            
            # Traiter chaque segment et ajouter la traduction
            print("Traduction des segments...")
            for i, segment in enumerate(tqdm(data['segments'], desc="Segments")):
                text = segment.get('text', '').strip()
                if text:
                    translation_info = self.vocabulary_processor.get_basic_translation(text)
                    segment['translation'] = translation_info['fr']
            
            # Extraire le vocabulaire
            print("Extraction du vocabulaire...")
            vocabulary_words = self.vocabulary_processor.extract_vocabulary(data['segments'])
            
            # Traiter le vocabulaire avec prononciations
            print(f"Traitement de {len(vocabulary_words)} mots du vocabulaire...")
            vocabulary_entries = self.vocabulary_processor.process_vocabulary_batch(vocabulary_words)
            
            # Statistiques sur les prononciations
            pronunciation_stats = {
                'total_words': len(vocabulary_entries),
                'pronunciations_found': len([v for v in vocabulary_entries if v.get('pronunciation')]),
                'pronunciations_cached': len([v for v in vocabulary_entries if v.get('pronunciation', {}).get('status') == 'cached']),
                'pronunciations_downloaded': len([v for v in vocabulary_entries if v.get('pronunciation', {}).get('status') == 'downloaded'])
            }
            
            # Ajouter les données au JSON
            data['vocabulary'] = vocabulary_entries
            data['vocabulary_stats'] = {
                'total_words': len(vocabulary_entries),
                'levels': {
                    'beginner': len([w for w in vocabulary_entries if w['level'] == 'beginner']),
                    'intermediate': len([w for w in vocabulary_entries if w['level'] == 'intermediate']),
                    'advanced': len([w for w in vocabulary_entries if w['level'] == 'advanced'])
                },
                'pronunciations': pronunciation_stats
            }
            
            # Sauvegarder le fichier modifié
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print("✓ Traitement du vocabulaire terminé")
            print(f"✓ Prononciations: {pronunciation_stats['pronunciations_found']}/{pronunciation_stats['total_words']} trouvées")
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
        
        # Étape 2: Traitement du vocabulaire avec prononciations
        print("\n=== ÉTAPE 2: TRAITEMENT DU VOCABULAIRE + PRONONCIATIONS ===")
        if not self.process_transcription_file(json_path, output_dir):
            return False
        
        print(f"\n✓ Traitement complet terminé!")
        print(f"Fichier de sortie: {json_path}")
        print(f"Dossier prononciations: {Path(output_dir) / 'pronunciations'}")
        return True

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Transcrit et analyse un fichier audio en allemand avec récupération des prononciations",
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
        "--pronunciation-delay",
        type=float,
        default=CONFIG['pronunciation_delay'],
        help=f"Délai entre les requêtes de prononciation en secondes (défaut: {CONFIG['pronunciation_delay']})"
    )
    
    args = parser.parse_args()
    
    # Mettre à jour la configuration
    CONFIG['max_vocabulary_size'] = args.max_vocab
    CONFIG['whisper_model'] = args.model
    CONFIG['pronunciation_delay'] = args.pronunciation_delay
    
    # Traitement
    processor = TranscriptionProcessor()
    success = processor.process_audio_file(args.audio_file, args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()