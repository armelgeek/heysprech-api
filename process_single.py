#!/usr/bin/env python3
import os
import re
import sys
import subprocess
import argparse
import requests
from bs4 import BeautifulSoup
import urllib.parse
from urllib.parse import urljoin
import json
import random
import time
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
    'french_gpt_model': "dbddv01/gpt2-french-small",
    'max_vocabulary_size': 100,  # Limiter le nombre de mots à traiter
    'min_word_length': 2,
    'max_examples_per_word': 3,
    'max_exercises_per_type': 2
}

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

def download_audio_file(audio_url, filename=None):
    """
    Télécharge un fichier audio
    """
    if not filename:
        filename = audio_url.split('/')[-1]
        if not any(ext in filename.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
            filename += '.mp3'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(audio_url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Fichier téléchargé: {filename}")
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return False

class VocabularyProcessor:
    """Classe pour gérer le traitement du vocabulaire et les analyses lexicales"""
    
    def __init__(self, models: Dict):
        self.models = models
        self.processed_words = set()
        self.pronunciation_dir = Path("pronunciations")
        self.pronunciation_dir.mkdir(exist_ok=True)
        
    def scrape_audio_tags(self, word: str) -> List[Dict]:
        """Scrape les balises audio pour un mot"""
        url = f"https://howpronounce.com/german/{word}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        print(f"Recherche de prononciation pour : {word}")
        
        try:
            # Ajout d'un délai aléatoire entre 1 et 3 secondes
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parser le HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            audio_sources = []
            
            # 1. Rechercher les balises audio directes
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
                
                for source in audio.find_all('source'):
                    if source.get('src'):
                        src = source.get('src')
                        full_url = urljoin(url, src)
                        audio_sources.append({
                            'type': 'source_tag',
                            'url': full_url,
                            'original_src': src,
                            'mime_type': source.get('type', 'unknown')
                        })
            
            # 2. Rechercher dans le JavaScript
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string:
                    # Chercher des URLs audio
                    audio_urls = re.findall(r'["\']([^"\']*\.(?:mp3|wav|ogg|m4a)[^"\']*)["\']', script.string)
                    for audio_url in audio_urls:
                        full_url = urljoin(url, audio_url)
                        audio_sources.append({
                            'type': 'js_embedded',
                            'url': full_url,
                            'original_src': audio_url
                        })
            
            # 3. Rechercher les attributs data-*
            elements_with_data = soup.find_all(lambda tag: any(attr.startswith('data-') for attr in tag.attrs))
            for element in elements_with_data:
                for attr, value in element.attrs.items():
                    if attr.startswith('data-') and isinstance(value, str):
                        if any(ext in value.lower() for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
                            full_url = urljoin(url, value)
                            audio_sources.append({
                                'type': f'data_attribute_{attr}',
                                'url': full_url,
                                'original_src': value
                            })
            
            print(f"✓ {len(audio_sources)} sources audio trouvées pour '{word}'")
            return audio_sources
            
        except Exception as e:
            print(f"✗ Erreur lors du scraping pour '{word}': {e}")
            return []
    
    def download_audio_file(self, audio_url: str, filename: str) -> bool:
        """Télécharge un fichier audio de prononciation"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(audio_url, headers=headers, stream=True)
            response.raise_for_status()
            
            filepath = self.pronunciation_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✓ Prononciation téléchargée : {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors du téléchargement : {e}")
            return False
    
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
        tokenizer_gpt_de, model_gpt_de = self.models['gpt_de']
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
        
        # 1. Exercice DE -> FR
        de_fr_exercise = self._create_exercise_variant(
            word=word,
            correct_answer=translation,
            question_type="de_to_fr",
            question_de=f"Welches französische Wort bedeutet '{word}'?",
            question_fr=f"Quelle est la traduction française de '{word}'?",
            distractor_language="fr"
        )
        
        # 2. Exercice FR -> DE
        fr_de_exercise = self._create_exercise_variant(
            word=translation,
            correct_answer=word,
            question_type="fr_to_de", 
            question_de=f"Welches deutsche Wort bedeutet '{translation}'?",
            question_fr=f"Quel est le mot allemand pour '{translation}'?",
            distractor_language="de"
        )
        
        return {
            'type': 'multiple_choice_pair',
            'de_to_fr': de_fr_exercise,
            'fr_to_de': fr_de_exercise,
            'level': self.determine_word_level(word)
        }

    def _create_exercise_variant(self, word: str, correct_answer: str, question_type: str,
                            question_de: str, question_fr: str, distractor_language: str) -> Dict:
        """Crée un exercice QCM dans un sens de traduction spécifique"""
        
        try:
            # Générer les distracteurs (mauvaises réponses)
            wrong_options = self._generate_distractors(
                word=word, 
                correct_answer=correct_answer,
                target_language=distractor_language,
                count=3
            )
            
            # Créer la liste des options avec la bonne réponse
            options = [correct_answer] + wrong_options
            random.shuffle(options)
            
            return {
                'type': 'multiple_choice',
                'question_type': question_type,
                'question': {
                    'de': question_de,
                    'fr': question_fr
                },
                'word_to_translate': word,
                'options': options,
                'correct_answer': correct_answer,
                'level': self.determine_word_level(word if question_type == "de_to_fr" else correct_answer)
            }
            
        except Exception as e:
            print(f"Erreur lors de la génération de l'exercice pour '{word}': {e}")
            # Fallback avec des options simples
            return self._create_fallback_exercise(word, correct_answer, question_de, question_fr, question_type)

    def _generate_distractors(self, word: str, correct_answer: str, target_language: str, count: int = 3) -> List[str]:
        """Génère des distracteurs sémantiquement proches de manière dynamique"""
        distractors = []
        tokenizer_gpt_de, model_gpt_de = self.models['gpt_de']
        tokenizer_gpt_fr, model_gpt_fr = self.models['gpt_fr']
        tokenizer_de_fr, model_de_fr = self.models['de_fr']
        
        # Calculer la longueur cible
        target_length = len(correct_answer)
        min_length = max(2, target_length - 2)
        max_length = target_length + 3
    
        def is_valid_distractor(candidate: str) -> bool:
            """Vérifie si un candidat est un bon distracteur"""
            candidate_clean = candidate.strip().lower()
            correct_clean = correct_answer.strip().lower()
            
            return (
                min_length <= len(candidate_clean) <= max_length and
                candidate_clean != correct_clean and
                candidate_clean != word.lower() and
                candidate_clean not in [d.lower() for d in distractors] and
                len(candidate_clean) >= 2 and
                re.match(r'^[a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇäöüßÄÖÜ]+$', candidate_clean)
            )
        
        # Stratégie 1: Génération contextuelle avec GPT
        if target_language == "fr":
            context_prompts = [
                f"Mots français de même type que '{correct_answer}': {correct_answer}, ",
                f"Synonymes français de '{correct_answer}': {correct_answer}, ",
                f"Mots français proches: {correct_answer}, ",
                f"Vocabulaire français similaire à '{correct_answer}': "
            ]
            tokenizer_gpt = tokenizer_gpt_fr
            model_gpt = model_gpt_fr
        else:
            context_prompts = [
                f"Deutsche Wörter vom gleichen Typ wie '{correct_answer}': {correct_answer}, ",
                f"Deutsche Synonyme von '{correct_answer}': {correct_answer}, ",
                f"Ähnliche deutsche Wörter: {correct_answer}, ",
                f"Deutscher Wortschatz ähnlich wie '{correct_answer}': "
            ]
            tokenizer_gpt = tokenizer_gpt_de
            model_gpt = model_gpt_de
        
        for prompt in context_prompts:
            if len(distractors) >= count:
                break
            
            try:
                inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True, truncation=True, max_length=50)
                outputs = model_gpt.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 15,
                    num_return_sequences=2,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer_gpt.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                
                for output in outputs:
                    generated_text = tokenizer_gpt.decode(output, skip_special_tokens=True)
                    new_part = generated_text.replace(prompt, "").strip()
                    
                    # Extraire les mots selon la langue
                    if target_language == "fr":
                        words = re.findall(r'\b[a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ]+\b', new_part)
                    else:
                        words = re.findall(r'\b[a-zäöüßA-ZÄÖÜ]+\b', new_part)
                    
                    for extracted_word in words[:3]:  # Limiter à 3 mots par génération
                        clean_word = extracted_word.strip()
                        if is_valid_distractor(clean_word) and len(distractors) < count:
                            # Adapter la casse au mot correct
                            if correct_answer[0].isupper():
                                clean_word = clean_word.capitalize()
                            else:
                                clean_word = clean_word.lower()
                            distractors.append(clean_word)
                            
            except Exception as e:
                print(f"Erreur génération GPT: {e}")
                continue
        
        # Stratégie 2: Génération par traduction inverse si pas assez de distracteurs
        if len(distractors) < count:
            try:
                # Générer des mots proches du mot source et les traduire
                if target_language == "fr":
                    # Générer des mots allemands proches du mot source
                    source_prompts = [
                        f"Deutsche Wörter: {word}, ",
                        f"Ähnliche Wörter wie {word}: {word}, "
                    ]
                    for prompt in source_prompts:
                        if len(distractors) >= count:
                            break
                        
                        inputs = tokenizer_gpt_de(prompt, return_tensors="pt", padding=True, truncation=True)
                        outputs = model_gpt_de.generate(
                            inputs.input_ids,
                            max_length=inputs.input_ids.shape[1] + 10,
                            num_return_sequences=2,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer_gpt_de.eos_token_id
                        )
                        
                        for output in outputs:
                            generated_text = tokenizer_gpt_de.decode(output, skip_special_tokens=True)
                            new_part = generated_text.replace(prompt, "").strip()
                            german_words = re.findall(r'\b[a-zäöüßA-ZÄÖÜ]+\b', new_part)
                            
                            for german_word in german_words[:2]:
                                if german_word.lower() != word.lower():
                                    # Traduire ce mot allemand en français
                                    try:
                                        translation_input = tokenizer_de_fr(german_word, return_tensors="pt")
                                        translation_output = model_de_fr.generate(
                                            translation_input.input_ids,
                                            max_length=20,
                                            num_beams=2,
                                            temperature=0.3
                                        )
                                        french_distractor = tokenizer_de_fr.decode(
                                            translation_output[0], skip_special_tokens=True
                                        ).strip()
                                        
                                        if is_valid_distractor(french_distractor) and len(distractors) < count:
                                            distractors.append(french_distractor)
                                            
                                    except Exception:
                                        continue
                else:
                    # Générer des mots français et les traduire en allemand
                    source_prompts = [
                        f"Mots français: {correct_answer}, ",
                        f"Mots similaires à {correct_answer}: {correct_answer}, "
                    ]
                    # Note: Pour cette stratégie, nous aurions besoin d'un modèle FR->DE
                    # qui n'est pas disponible dans le code actuel
                    pass
                    
            except Exception as e:
                print(f"Erreur traduction inverse: {e}")
        
        # Stratégie 3: Variations morphologiques générées
        if len(distractors) < count:
            base = correct_answer.lower()
            
            if target_language == "fr":
                # Variations morphologiques françaises générées
                morphological_patterns = [
                    (lambda x: x + "s", "pluriel"),
                    (lambda x: x + "e", "féminin"),
                    (lambda x: x + "er", "infinitif"),
                    (lambda x: x + "eur", "agent"),
                    (lambda x: x + "tion" if len(x) <= 6 else x, "nominalisation"),
                    (lambda x: "dé" + x if len(x) <= 6 else x, "préfixe"),
                    (lambda x: x + "ment" if len(x) <= 5 else x, "adverbe"),
                    (lambda x: x + "age" if len(x) <= 5 else x, "action"),
                    (lambda x: x + "ible" if len(x) <= 4 else x, "adjectif"),
                    (lambda x: x + "iste" if len(x) <= 5 else x, "spécialiste")
                ]
            else:
                # Variations morphologiques allemandes générées
                morphological_patterns = [
                    (lambda x: x + "en", "pluriel/infinitif"),
                    (lambda x: x + "er", "agent/comparatif"),
                    (lambda x: x + "ung", "nominalisation"),
                    (lambda x: x + "heit" if len(x) <= 5 else x, "qualité"),
                    (lambda x: "ge" + x if len(x) <= 6 else x, "participe"),
                    (lambda x: x + "lich" if len(x) <= 4 else x, "adjectif"),
                    (lambda x: x + "schaft" if len(x) <= 4 else x, "collectif"),
                    (lambda x: x + "bar" if len(x) <= 5 else x, "possibilité"),
                    (lambda x: x + "los" if len(x) <= 5 else x, "privation"),
                    (lambda x: "un" + x if len(x) <= 6 else x, "négation")
                ]
            
            for pattern_func, pattern_type in morphological_patterns:
                if len(distractors) >= count:
                    break
                
                try:
                    variant = pattern_func(base)
                    if is_valid_distractor(variant):
                        # Adapter la casse
                        if correct_answer[0].isupper():
                            variant = variant.capitalize()
                        distractors.append(variant)
                except Exception:
                    continue
        
        # Stratégie 4: Génération par rhyme/assonance si encore insuffisant
        if len(distractors) < count:
            # Générer des mots qui sonnent similaires
            base = correct_answer.lower()
            
            if len(base) >= 3:
                # Variations phonétiques
                phonetic_variants = []
                
                # Changer la première lettre
                for letter in 'abcdefghijklmnopqrstuvwxyz':
                    if letter != base[0]:
                        variant = letter + base[1:]
                        phonetic_variants.append(variant)
                
                # Changer la dernière lettre
                for letter in 'abcdefghijklmnopqrstuvwxyz':
                    if letter != base[-1]:
                        variant = base[:-1] + letter
                        phonetic_variants.append(variant)
                
                # Changer une lettre au milieu
                if len(base) > 3:
                    mid_pos = len(base) // 2
                    for letter in 'abcdefghijklmnopqrstuvwxyz':
                        if letter != base[mid_pos]:
                            variant = base[:mid_pos] + letter + base[mid_pos+1:]
                            phonetic_variants.append(variant)
                
                # Tester les variantes phonétiques
                for variant in phonetic_variants:
                    if len(distractors) >= count:
                        break
                    if is_valid_distractor(variant):
                        if correct_answer[0].isupper():
                            variant = variant.capitalize()
                        distractors.append(variant)
        
        return distractors[:count]
    def _create_fallback_exercise(self, word: str, correct_answer: str, question_de: str, 
                                question_fr: str, question_type: str) -> Dict:
        """Crée un exercice de fallback avec des options prédéfinies"""
        
        if question_type == "de_to_fr":
            # Distracteurs français simples
            wrong_options = [
                correct_answer + "er",
                "un " + correct_answer, 
                correct_answer + "tion"
            ]
        else:
            # Distracteurs allemands simples  
            wrong_options = [
                correct_answer + "en",
                "der " + correct_answer,
                correct_answer + "ung"
            ]
        
        options = [correct_answer] + wrong_options
        random.shuffle(options)
        
        return {
            'type': 'multiple_choice',
            'question_type': question_type,
            'question': {
                'de': question_de,
                'fr': question_fr
            },
            'word_to_translate': word,
            'options': options,
            'correct_answer': correct_answer,
            'level': self.determine_word_level(word if question_type == "de_to_fr" else correct_answer),
            'fallback': True
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
        # Traductions et exercices
        translation_info = self.get_basic_translation(word)
        example = self.generate_simple_example(word)
        exercise = self.create_simple_exercise(word, translation_info['fr'])
        level = self.determine_word_level(word)
        
        # Récupération de la prononciation
        pronunciation_info = {'available': False, 'file': None}
        audio_sources = self.scrape_audio_tags(word)
        
        if audio_sources:
            # Tenter de télécharger le premier fichier audio trouvé
            first_audio = audio_sources[0]
            audio_filename = f"{word}_pronunciation.mp3"
            
            if self.download_audio_file(first_audio['url'], audio_filename):
                pronunciation_info = {
                    'available': True,
                    'file': str(self.pronunciation_dir / audio_filename)
                }
        
        # Construire le résultat
        result = {
            'word': word,
            'level': level,
            'translation': translation_info,
            'example': example,
            'exercise': exercise,
            'pronunciation': pronunciation_info,
            'processed_at': self.get_timestamp()
        }
        
        return result
    
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
            tokenizer_gpt_de = AutoTokenizer.from_pretrained(CONFIG['german_gpt_model'])
            model_gpt_de = GPT2LMHeadModel.from_pretrained(CONFIG['german_gpt_model'])
            
            # Modèle GPT français
            print("→ Chargement du modèle GPT français...")
            tokenizer_gpt_fr = AutoTokenizer.from_pretrained(CONFIG['french_gpt_model'])
            model_gpt_fr = GPT2LMHeadModel.from_pretrained(CONFIG['french_gpt_model'])
            
            # Configuration des tokenizers GPT
            if tokenizer_gpt_de.pad_token is None:
                tokenizer_gpt_de.pad_token = tokenizer_gpt_de.eos_token
            if tokenizer_gpt_fr.pad_token is None:
                tokenizer_gpt_fr.pad_token = tokenizer_gpt_fr.eos_token
            
            self.models = {
                'de_fr': (tokenizer_de_fr, model_de_fr),
                'gpt_de': (tokenizer_gpt_de, model_gpt_de),
                'gpt_fr': (tokenizer_gpt_fr, model_gpt_fr)
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