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
        """Génère des distracteurs de la même famille sémantique/contextuelle"""
        distractors = []
        tokenizer_gpt_de, model_gpt_de = self.models['gpt_de']
        tokenizer_gpt_fr, model_gpt_fr = self.models['gpt_fr']
        
        def is_valid_distractor(candidate: str) -> bool:
            """Vérifie si un candidat est un bon distracteur"""
            candidate_clean = candidate.strip().lower()
            correct_clean = correct_answer.strip().lower()
            
            return (
                len(candidate_clean) >= 2 and
                len(candidate_clean) <= 25 and  # Plus flexible pour les expressions
                candidate_clean != correct_clean and
                candidate_clean != word.lower() and
                candidate_clean not in [d.lower() for d in distractors] and
                not candidate_clean.startswith(correct_clean[:3]) if len(correct_clean) > 3 else True
            )
        
        # Stratégie 1: Génération contextuelle - expressions de la même famille
        if target_language == "fr":
            context_prompts = [
                f"Expressions françaises dans le même contexte que '{correct_answer}': {correct_answer}, ",
                f"Phrases similaires à '{correct_answer}': {correct_answer}, ",
                f"Expressions de politesse comme '{correct_answer}': {correct_answer}, ",
                f"Dans la même situation on dit '{correct_answer}' ou ",
                f"Alternatives à '{correct_answer}': {correct_answer}, ",
                f"Expressions équivalentes: {correct_answer}, "
            ]
            tokenizer_gpt = tokenizer_gpt_fr
            model_gpt = model_gpt_fr
            # Patterns pour extraire les expressions françaises
            extraction_pattern = r'(?:[a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ]+(?:\s+[a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ]+)*)'
        else:
            context_prompts = [
                f"Deutsche Ausdrücke im gleichen Kontext wie '{correct_answer}': {correct_answer}, ",
                f"Ähnliche Phrasen wie '{correct_answer}': {correct_answer}, ",
                f"Höflichkeitsformen wie '{correct_answer}': {correct_answer}, ",
                f"In derselben Situation sagt man '{correct_answer}' oder ",
                f"Alternativen zu '{correct_answer}': {correct_answer}, ",
                f"Gleichwertige Ausdrücke: {correct_answer}, "
            ]
            tokenizer_gpt = tokenizer_gpt_de
            model_gpt = model_gpt_de
            # Patterns pour extraire les expressions allemandes
            extraction_pattern = r'(?:[a-zäöüßA-ZÄÖÜ]+(?:\s+[a-zäöüßA-ZÄÖÜ]+)*)'
        
        for prompt in context_prompts:
            if len(distractors) >= count:
                break
            
            try:
                inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True, truncation=True, max_length=60)
                outputs = model_gpt.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 20,
                    num_return_sequences=3,
                    temperature=0.9,  # Plus créatif pour les expressions
                    do_sample=True,
                    pad_token_id=tokenizer_gpt.eos_token_id,
                    top_p=0.95,
                    repetition_penalty=1.3
                )
                
                for output in outputs:
                    generated_text = tokenizer_gpt.decode(output, skip_special_tokens=True)
                    new_part = generated_text.replace(prompt, "").strip()
                    
                    # Extraire les expressions (pas juste les mots isolés)
                    # Diviser par des séparateurs communs
                    potential_expressions = re.split(r'[,;.!?\n]', new_part)
                    
                    for expr in potential_expressions[:4]:  # Limiter à 4 expressions par génération
                        expr = expr.strip()
                        if expr and is_valid_distractor(expr):
                            # Nettoyer l'expression
                            expr = re.sub(r'^[^\w]*|[^\w]*$', '', expr)  # Enlever ponctuation début/fin
                            expr = re.sub(r'\s+', ' ', expr)  # Normaliser espaces
                            
                            if expr and len(distractors) < count:
                                distractors.append(expr)
                                
            except Exception as e:
                print(f"Erreur génération contextuelle: {e}")
                continue
        
        # Stratégie 2: Génération par analogie contextuelle
        if len(distractors) < count:
            try:
                # Analyser le type d'expression pour générer des analogies
                expression_type = self._analyze_expression_type(correct_answer, target_language)
                
                if target_language == "fr":
                    analogy_prompts = self._get_french_analogy_prompts(expression_type, correct_answer)
                else:
                    analogy_prompts = self._get_german_analogy_prompts(expression_type, correct_answer)
                
                for prompt in analogy_prompts:
                    if len(distractors) >= count:
                        break
                    
                    inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True, truncation=True)
                    outputs = model_gpt.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 15,
                        num_return_sequences=2,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer_gpt.eos_token_id
                    )
                    
                    for output in outputs:
                        generated_text = tokenizer_gpt.decode(output, skip_special_tokens=True)
                        new_part = generated_text.replace(prompt, "").strip()
                        
                        # Extraire la première expression générée
                        first_expr = re.split(r'[,;.!?\n]', new_part)[0].strip()
                        if first_expr and is_valid_distractor(first_expr):
                            first_expr = re.sub(r'^[^\w]*|[^\w]*$', '', first_expr)
                            if first_expr and len(distractors) < count:
                                distractors.append(first_expr)
                                
            except Exception as e:
                print(f"Erreur génération par analogie: {e}")
        
        # Stratégie 3: Templates basés sur le type d'expression
        if len(distractors) < count:
            template_distractors = self._generate_template_distractors(
                correct_answer, target_language, count - len(distractors)
            )
            for template_dist in template_distractors:
                if is_valid_distractor(template_dist) and len(distractors) < count:
                    distractors.append(template_dist)
        
        return distractors[:count]

    def _analyze_expression_type(self, expression: str, language: str) -> str:
        """Analyse le type d'expression pour générer des analogies appropriées"""
        expr_lower = expression.lower().strip()
        
        if language == "fr":
            if any(word in expr_lower for word in ["à vous", "votre", "vous"]):
                return "interpellation_politesse"
            elif any(word in expr_lower for word in ["merci", "s'il vous plaît", "excusez"]):
                return "politesse_formelle"
            elif any(word in expr_lower for word in ["bonjour", "bonsoir", "salut"]):
                return "salutation"
            elif any(word in expr_lower for word in ["oui", "non", "peut-être"]):
                return "reponse_simple"
            elif any(word in expr_lower for word in ["comment", "où", "quand", "pourquoi"]):
                return "interrogation"
            elif any(word in expr_lower for word in ["il faut", "il est", "c'est"]):
                return "expression_impersonnelle"
        else:  # allemand
            if any(word in expr_lower for word in ["sie", "ihr", "euch"]):
                return "interpellation_politesse"
            elif any(word in expr_lower for word in ["danke", "bitte", "entschuldigung"]):
                return "politesse_formelle"
            elif any(word in expr_lower for word in ["hallo", "guten", "tschüss"]):
                return "salutation"
            elif any(word in expr_lower for word in ["ja", "nein", "vielleicht"]):
                return "reponse_simple"
            elif any(word in expr_lower for word in ["wie", "wo", "wann", "warum"]):
                return "interrogation"
            elif any(word in expr_lower for word in ["es ist", "das ist", "man"]):
                return "expression_impersonnelle"
        
        return "general"

    def _get_french_analogy_prompts(self, expression_type: str, original: str) -> List[str]:
        """Génère des prompts d'analogie pour le français selon le type d'expression"""
        prompts_by_type = {
            "interpellation_politesse": [
                "Façons polies de s'adresser à quelqu'un: ",
                "Expressions pour interpeller poliment: ",
                "Formules de courtoisie: "
            ],
            "politesse_formelle": [
                "Expressions de politesse française: ",
                "Formules de courtoisie: ",
                "Phrases polies courantes: "
            ],
            "salutation": [
                "Façons de saluer en français: ",
                "Formules de salutation: ",
                "Expressions pour dire bonjour: "
            ],
            "reponse_simple": [
                "Réponses courtes en français: ",
                "Façons de répondre: ",
                "Expressions de confirmation: "
            ],
            "interrogation": [
                "Questions courantes en français: ",
                "Façons de demander: ",
                "Expressions interrogatives: "
            ],
            "expression_impersonnelle": [
                "Expressions impersonnelles françaises: ",
                "Tournures avec il faut, il est: ",
                "Constructions impersonnelles: "
            ],
            "general": [
                f"Expressions similaires à '{original}': ",
                f"Alternatives à '{original}': ",
                "Expressions courantes: "
            ]
        }
        return prompts_by_type.get(expression_type, prompts_by_type["general"])

    def _get_german_analogy_prompts(self, expression_type: str, original: str) -> List[str]:
        """Génère des prompts d'analogie pour l'allemand selon le type d'expression"""
        prompts_by_type = {
            "interpellation_politesse": [
                "Höfliche Anreden auf Deutsch: ",
                "Ausdrücke für höfliche Ansprache: ",
                "Höflichkeitsformen: "
            ],
            "politesse_formelle": [
                "Deutsche Höflichkeitsausdrücke: ",
                "Höflichkeitsformeln: ",
                "Höfliche Redewendungen: "
            ],
            "salutation": [
                "Deutsche Begrüßungen: ",
                "Begrüßungsformeln: ",
                "Ausdrücke zum Grüßen: "
            ],
            "reponse_simple": [
                "Kurze Antworten auf Deutsch: ",
                "Antwortmöglichkeiten: ",
                "Bestätigungsausdrücke: "
            ],
            "interrogation": [
                "Häufige deutsche Fragen: ",
                "Frageausdrücke: ",
                "Interrogativformen: "
            ],
            "expression_impersonnelle": [
                "Unpersönliche deutsche Ausdrücke: ",
                "Konstruktionen mit es ist, man: ",
                "Unpersönliche Wendungen: "
            ],
            "general": [
                f"Ähnliche Ausdrücke wie '{original}': ",
                f"Alternativen zu '{original}': ",
                "Gängige Ausdrücke: "
            ]
        }
        return prompts_by_type.get(expression_type, prompts_by_type["general"])

    def _generate_template_distractors(self, original: str, language: str, needed_count: int) -> List[str]:
        """Génère des distracteurs basés sur des templates selon le type d'expression"""
        templates = []
        
        if language == "fr":
            # Templates français par contexte
            if "vous" in original.lower():
                templates = ["à moi", "à lui", "à elle", "à nous", "à toi", "chez vous", "avec vous", "pour vous"]
            elif "merci" in original.lower():
                templates = ["de rien", "je vous en prie", "avec plaisir", "tout le plaisir", "volontiers"]
            elif "bonjour" in original.lower():
                templates = ["bonsoir", "salut", "bonne journée", "au revoir", "à bientôt"]
            elif "oui" in original.lower():
                templates = ["non", "peut-être", "certainement", "bien sûr", "d'accord"]
            else:
                templates = ["très bien", "parfait", "exactement", "tout à fait", "absolument", "en effet"]
        else:
            # Templates allemands par contexte
            if "sie" in original.lower() or "ihr" in original.lower():
                templates = ["zu mir", "zu ihm", "zu ihr", "zu uns", "bei Ihnen", "mit Ihnen", "für Sie"]
            elif "danke" in original.lower():
                templates = ["bitte", "gern geschehen", "keine Ursache", "mit Vergnügen"]
            elif "hallo" in original.lower() or "guten" in original.lower():
                templates = ["tschüss", "auf Wiedersehen", "bis bald", "schönen Tag"]
            elif "ja" in original.lower():
                templates = ["nein", "vielleicht", "bestimmt", "natürlich", "einverstanden"]
            else:
                templates = ["sehr gut", "perfekt", "genau", "völlig richtig", "absolut", "in der Tat"]
        
        return templates[:needed_count]
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