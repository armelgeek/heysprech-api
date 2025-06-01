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
    AutoTokenizer,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    pipeline
)

from tqdm import tqdm


def get_output_folder(audio_path: str) -> Path:
    """Crée et retourne le dossier de sortie pour un fichier audio"""
    audio_name = Path(audio_path).stem
    output_dir = Path("/app") / CONFIG['language'] / audio_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# Configuration globale
CONFIG = {
    'whisper_model': "base",
    'language': "de",  # langue source par défaut
    'output_format': "json",
    'language_pairs': {
        'de-fr': {
            'translation_model_path': "./opus-mt-de-fr",
            'source_gpt_model': "benjamin/gpt2-wechsel-german",
            'target_gpt_model': "dbddv01/gpt2-french-small",
        },
        'en-fr': {
            'translation_model_path': "Helsinki-NLP/opus-mt-en-fr",
            'source_gpt_model': "gpt2",
            'target_gpt_model': "dbddv01/gpt2-french-small",
        },
        # Ajoutez d'autres paires de langues ici
    },
    'max_vocabulary_size': 100,
    'min_word_length': 2,
    'max_examples_per_word': 3,
    'max_exercises_per_type': 2
}

# Mapping des codes de langue vers les noms complets
LANGUAGE_NAMES = {
    'de': 'German',
    'fr': 'French',
    'en': 'English',
    # Ajoutez d'autres langues selon vos besoins
}

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)
class TranslationRefiner:
    """Classe pour raffiner et reformuler les traductions"""
    
    def __init__(self):
        self.refiner_model = None
        self.refiner_tokenizer = None
        self.correction_pipeline = None
        self.load_refinement_models()
    
    def load_refinement_models(self):
        """Charge les modèles de reformulation"""
        try:
            # Option 1: Modèle T5 français pour reformulation
            print("→ Chargement du modèle de reformulation T5...")
            self.refiner_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.refiner_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            
            # Option 2: Pipeline de correction grammaticale
            print("→ Chargement du pipeline de correction...")
            self.correction_pipeline = pipeline(
                "text2text-generation",
                model="pszemraj/flan-t5-base-grammar-synthesis",
                tokenizer="pszemraj/flan-t5-base-grammar-synthesis"
            )
            
        except Exception as e:
            print(f"Erreur chargement modèles de reformulation: {e}")
    
    def refine_translation_with_t5(self, raw_translation: str, source_word: str) -> str:
        """Utilise T5 pour reformuler la traduction"""
        if not self.refiner_model:
            return raw_translation
        
        try:
            # Créer un prompt de reformulation
            prompt = f"reformulate: {raw_translation}"
            
            inputs = self.refiner_tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=64,
                truncation=True
            )
            
            outputs = self.refiner_model.generate(
                inputs.input_ids,
                max_length=32,
                num_beams=3,
                temperature=0.3,
                do_sample=False,
                early_stopping=True
            )
            
            refined = self.refiner_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Vérifier que c'est cohérent
            if refined and len(refined.split()) <= 3 and refined != source_word:
                return refined.strip()
            
        except Exception as e:
            print(f"Erreur reformulation T5: {e}")
        
        return raw_translation
    
    def refine_with_grammar_correction(self, raw_translation: str) -> str:
        """Utilise un modèle de correction grammaticale"""
        if not self.correction_pipeline:
            return raw_translation
        
        try:
            # Créer un prompt de correction
            prompt = f"correct: {raw_translation}"
            
            result = self.correction_pipeline(
                prompt,
                max_length=50,
                num_beams=3,
                temperature=0.2
            )
            
            corrected = result[0]['generated_text'].strip()
            
            # Vérifier la cohérence
            if corrected and len(corrected.split()) <= 3:
                return corrected
                
        except Exception as e:
            print(f"Erreur correction grammaticale: {e}")
        
        return raw_translation
    
    def contextual_refinement(self, word: str, raw_translation: str, context: str = None) -> str:
        """Reformulation contextuelle avec prompts intelligents"""
        refinement_prompts = [
            f"Translate '{word}' to French:",
            f"What is '{word}' in French?",
            f"French word for '{word}':",
            f"'{word}' means in French:"
        ]
        
        # Utiliser le modèle GPT français pour reformuler
        if hasattr(self, 'gpt_fr_model'):
            tokenizer_gpt_fr, model_gpt_fr = self.gpt_fr_model
            
            for prompt in refinement_prompts:
                try:
                    inputs = tokenizer_gpt_fr(prompt, return_tensors="pt", padding=True)
                    outputs = model_gpt_fr.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 8,
                        num_beams=3,
                        temperature=0.2,
                        do_sample=False,
                        pad_token_id=tokenizer_gpt_fr.eos_token_id
                    )
                    
                    generated = tokenizer_gpt_fr.decode(outputs[0], skip_special_tokens=True)
                    # Extraire seulement la partie après le prompt
                    if prompt in generated:
                        refined = generated.replace(prompt, "").strip()
                        refined = refined.split()[0] if refined.split() else ""
                        
                        if (refined and refined.lower() != word.lower() 
                            and len(refined) >= 2 and len(refined) <= 15):
                            return refined
                            
                except Exception:
                    continue
        
        return raw_translation
    
    def rule_based_refinement(self, word: str, raw_translation: str) -> str:
        """Reformulation basée sur des règles linguistiques"""
        
        # Nettoyer la traduction brute
        translation = raw_translation.strip()
        
        # Règle 1: Supprimer le mot source s'il apparaît
        if translation.lower().startswith(word.lower()):
            translation = translation[len(word):].strip()
        
        # Règle 2: Supprimer les délimiteurs courants
        delimiters = [" - ", " : ", " → ", " = ", ", ", "  "]
        for delimiter in delimiters:
            if delimiter in translation:
                parts = translation.split(delimiter)
                # Prendre la partie qui n'est pas le mot original
                for part in parts:
                    clean_part = part.strip()
                    if clean_part.lower() != word.lower() and len(clean_part) >= 2:
                        translation = clean_part
                        break
        
        # Règle 3: Gérer les articles français
        if translation.startswith(("le ", "la ", "les ", "un ", "une ", "des ")):
            # Garder l'article si c'est un nom
            if self._is_likely_noun(word):
                pass  # Garder l'article
            else:
                # Supprimer l'article pour les autres types de mots
                translation = " ".join(translation.split()[1:])
        
        # Règle 4: Prendre le premier mot si multiple et cohérent
        words = translation.split()
        if len(words) > 1:
            first_word = words[0].strip('.,;:!?()[]{}')
            if (first_word.lower() != word.lower() 
                and len(first_word) >= 2 
                and not first_word.lower() in ['le', 'la', 'les', 'un', 'une', 'des']):
                translation = first_word
        
        # Règle 5: Validation finale
        translation = translation.strip('.,;:!?()[]{}"\'-')
        
        return translation if translation and translation.lower() != word.lower() else word
    
    def _is_likely_noun(self, word: str) -> bool:
        """Détermine si un mot allemand est probablement un nom"""
        # En allemand, les noms commencent par une majuscule
        return word[0].isupper() if word else False
    
    def ensemble_refinement(self, word: str, raw_translation: str) -> Dict[str, str]:
        """Combine plusieurs méthodes de reformulation pour le meilleur résultat"""
        
        candidates = []
        
        # Méthode 1: Règles linguistiques (rapide et fiable)
        rule_based = self.rule_based_refinement(word, raw_translation)
        candidates.append(("rule_based", rule_based))
        
        # Méthode 2: Reformulation contextuelle
        contextual = self.contextual_refinement(word, raw_translation)
        candidates.append(("contextual", contextual))
        
        # Méthode 3: T5 reformulation (si disponible)
        if self.refiner_model:
            t5_refined = self.refine_translation_with_t5(raw_translation, word)
            candidates.append(("t5", t5_refined))
        
        # Méthode 4: Correction grammaticale (si disponible)
        if self.correction_pipeline:
            grammar_corrected = self.refine_with_grammar_correction(raw_translation)
            candidates.append(("grammar", grammar_corrected))
        
        # Sélectionner le meilleur candidat
        best_translation = self._select_best_candidate(word, candidates)
        
        return {
            'de': word,
            'fr': best_translation,
            'raw_translation': raw_translation,
            'candidates': {name: trans for name, trans in candidates},
            'method_used': self._get_winning_method(word, candidates, best_translation)
        }
    
    def _select_best_candidate(self, word: str, candidates: List[tuple]) -> str:
        """Sélectionne le meilleur candidat parmi les reformulations"""
        
        scored_candidates = []
        
        for method_name, translation in candidates:
            score = self._score_translation(word, translation)
            scored_candidates.append((score, translation, method_name))
        
        # Trier par score décroissant
        scored_candidates.sort(reverse=True)
        
        # Retourner la meilleure traduction
        return scored_candidates[0][1] if scored_candidates else word
    
    def _score_translation(self, word: str, translation: str) -> float:
        """Score une traduction selon plusieurs critères"""
        if not translation or translation.lower() == word.lower():
            return 0.0
        
        score = 1.0
        
        # Critère 1: Longueur appropriée (favoriser 3-12 caractères)
        length = len(translation)
        if 3 <= length <= 12:
            score += 1.0
        elif length < 3 or length > 20:
            score -= 0.5
        
        # Critère 2: Pas de chiffres ou caractères spéciaux étranges
        if re.match(r'^[a-zA-ZàâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ\s\-]+$', translation):
            score += 0.5
        else:
            score -= 1.0
        
        # Critère 3: Pas de répétition du mot original
        if word.lower() not in translation.lower():
            score += 0.5
        
        # Critère 4: Un seul mot est généralement préférable
        word_count = len(translation.split())
        if word_count == 1:
            score += 0.3
        elif word_count == 2:
            score += 0.1
        elif word_count > 3:
            score -= 0.3
        
        return score
    
    def _get_winning_method(self, word: str, candidates: List[tuple], best_translation: str) -> str:
        """Identifie quelle méthode a produit la meilleure traduction"""
        for method_name, translation in candidates:
            if translation == best_translation:
                return method_name
        return "unknown"

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
    
    def __init__(self, models: Dict, output_dir: str, source_lang: str = "de", target_lang: str = "fr"):
        self.models = models
        self.processed_words = set()
        self.output_dir = Path(output_dir)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.lang_pair = f"{source_lang}-{target_lang}"
        
        # Vérifier si la paire de langues est supportée
        if self.lang_pair not in CONFIG['language_pairs']:
            raise ValueError(f"Paire de langues non supportée: {self.lang_pair}")
        
        # Obtenir la configuration spécifique à la paire de langues
        self.lang_config = CONFIG['language_pairs'][self.lang_pair]

    def scrape_audio_tags(self, word: str) -> List[Dict]:
        """Scrape les balises audio pour un mot dans la langue source"""
        # Mapping des langues pour les URLs des sites de prononciation
        pronunciation_urls = {
            'de': f"https://howpronounce.com/german/{word}",
            'en': f"https://howpronounce.com/english/{word}",
            'fr': f"https://howpronounce.com/french/{word}",
            # Ajouter d'autres langues selon les besoins
        }

        # Vérifier si la langue source est supportée
        if self.source_lang not in pronunciation_urls:
            print(f"✗ Langue non supportée pour la prononciation: {self.source_lang}")
            return []

        url = pronunciation_urls[self.source_lang]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        print(f"Recherche de prononciation ({self.source_lang}) pour : {word}")
        
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
                        'original_src': src,
                        'language': self.source_lang
                    })
                
                for source in audio.find_all('source'):
                    if source.get('src'):
                        src = source.get('src')
                        full_url = urljoin(url, src)
                        audio_sources.append({
                            'type': 'source_tag',
                            'url': full_url,
                            'original_src': src,
                            'mime_type': source.get('type', 'unknown'),
                            'language': self.source_lang
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
                            'original_src': audio_url,
                            'language': self.source_lang
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
                                'original_src': value,
                                'language': self.source_lang
                            })
            
            print(f"✓ {len(audio_sources)} sources audio trouvées pour '{word}' ({self.source_lang})")
            return audio_sources
            
        except Exception as e:
            print(f"✗ Erreur lors du scraping pour '{word}' ({self.source_lang}): {e}")
            return []
    
    def download_audio_file(self, audio_url: str, filename: str) -> bool:
        """Télécharge un fichier audio de prononciation"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(audio_url, headers=headers, stream=True)
            response.raise_for_status()
            
            filepath = self.output_dir / filename
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
        tokenizer, model = self.models[self.lang_pair]
        
        try:
            # Encode with padding and return tensors
            inputs = tokenizer(word, return_tensors="pt", padding=True)
            
            # Generate translation with basic settings
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                num_beams=3,
                temperature=0.3,
                do_sample=False
            )
            
            # Decode the translation
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            return {
                self.source_lang: word,
                self.target_lang: translation if translation else word
            }
        except Exception as e:
            print(f"Erreur de traduction pour '{word}': {e}")
            return {self.source_lang: word, self.target_lang: word}

    def generate_simple_example(self, word: str) -> Dict[str, str]:
        """Génère un exemple simple d'utilisation"""
        source_tokenizer, source_model = self.models[f'gpt_{self.source_lang}']
        translator_tokenizer, translator_model = self.models[self.lang_pair]
        
        # Templates adaptés à la langue source
        templates = self._get_language_templates(word)
        
        try:
            # Choisir un template au hasard
            source_sentence = random.choice(templates)
            
            # Traduire vers la langue cible
            inputs = translator_tokenizer(source_sentence, return_tensors="pt", padding=True)
            outputs = translator_model.generate(
                inputs.input_ids,
                max_length=30,
                num_beams=3,
                temperature=0.3
            )
            target_sentence = translator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            return {
                self.source_lang: source_sentence,
                self.target_lang: target_sentence if target_sentence else f"Example with {word}"
            }
            
        except Exception as e:
            print(f"Erreur génération exemple pour '{word}': {e}")
            return {
                self.source_lang: f"This is {word}.",
                self.target_lang: f"C'est {word}."
            }

    def _get_language_templates(self, word: str) -> List[str]:
        """Retourne des templates adaptés à la langue source"""
        templates = {
            'de': [
                f"Das ist ein {word}.",
                f"Ich habe einen {word}.",
                f"Der {word} ist schön.",
                f"Wir brauchen {word}."
            ],
            'en': [
                f"This is a {word}.",
                f"I have a {word}.",
                f"The {word} is nice.",
                f"We need {word}."
            ],
            # Ajoutez d'autres langues selon vos besoins
        }
        
        return templates.get(self.source_lang, [f"Example with {word}."])
    
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

    def _generate_distractors(self, word: str, correct_answer: str, target_language: str, count: int = 5) -> List[str]:
        """Génère des distracteurs sémantiquement proches de manière dynamique"""
        distractors = []
        # Récupérer les modèles GPT pour les langues source et cible
        source_tokenizer, source_model = self.models[f'gpt_{self.source_lang}']
        target_tokenizer, target_model = self.models[f'gpt_{self.target_lang}']
        translator_tokenizer, translator_model = self.models[self.lang_pair]
        
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
        
        # Prompts adaptés à la langue source
        prompt_templates = {
            'de': [
                f"Wörter vom gleichen Typ wie '{word}': {word}, ",
                f"Synonyme von '{word}': {word}, ",
                f"Ähnliche Wörter: {word}, ",
                f"Wortschatz ähnlich wie '{word}': "
            ],
            'en': [
                f"Words similar to '{word}': {word}, ",
                f"Synonyms of '{word}': {word}, ",
                f"Related words to '{word}': {word}, ",
                f"Words in the same category as '{word}': "
            ],
            'fr': [
                f"Mots similaires à '{word}' : {word}, ",
                f"Synonymes de '{word}' : {word}, ",
                f"Mots apparentés à '{word}' : {word}, ",
                f"Vocabulaire proche de '{word}' : "
            ]
        }
        
        # Patterns morphologiques adaptés à la langue source
        morphological_patterns = {
            'de': [
                (lambda x: x + "en", "pluriel/infinitif"),
                (lambda x: x + "er", "agent/comparatif"),
                (lambda x: x + "ung", "nominalisation"),
                (lambda x: x + "heit" if len(x) <= 5 else x, "qualité"),
                (lambda x: "ge" + x if len(x) <= 6 else x, "participe")
            ],
            'en': [
                (lambda x: x + "s", "pluriel"),
                (lambda x: x + "ed", "participe passé"),
                (lambda x: x + "ing", "participe présent"),
                (lambda x: x + "er", "comparatif"),
                (lambda x: "un" + x if len(x) <= 6 else x, "négation")
            ],
            'fr': [
                (lambda x: x + "s", "pluriel"),
                (lambda x: x + "e", "féminin"),
                (lambda x: x + "ment", "adverbe"),
                (lambda x: x + "eur", "agent"),
                (lambda x: x + "ation", "nominalisation")
            ]
        }

        # Stratégie 1: Génération contextuelle avec GPT
        context_prompts = prompt_templates.get(self.source_lang, [f"Words similar to '{word}': "])
        
        if target_language == self.target_lang:
            # Générer des mots dans la langue source puis les traduire
            source_candidates = []
            for prompt in context_prompts:
                if len(source_candidates) >= count * 2:
                    break
                
                try:
                    inputs = source_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=50)
                    outputs = source_model.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 15,
                        num_return_sequences=2,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=source_tokenizer.eos_token_id,
                        top_p=0.9,
                        repetition_penalty=1.2
                    )
                    
                    for output in outputs:
                        generated_text = source_tokenizer.decode(output, skip_special_tokens=True)
                        new_part = generated_text.replace(prompt, "").strip()
                        
                        # Extraire les mots
                        words = re.findall(r'\b\w+\b', new_part)
                        
                        for extracted_word in words[:3]:
                            clean_word = extracted_word.strip().lower()
                            if (clean_word != word.lower() and 
                                len(clean_word) >= 2 and 
                                clean_word not in source_candidates):
                                source_candidates.append(clean_word)
                                
                except Exception as e:
                    print(f"Erreur génération GPT: {e}")
                    continue
            
            # Traduire les candidats vers la langue cible
            for source_word in source_candidates:
                if len(distractors) >= count:
                    break
                    
                try:
                    inputs = translator_tokenizer(source_word, return_tensors="pt", padding=True)
                    outputs = translator_model.generate(
                        inputs.input_ids,
                        max_length=20,
                        num_beams=3,
                        temperature=0.3,
                        do_sample=False
                    )
                    translation = translator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    
                    if is_valid_distractor(translation):
                        # Adapter la casse au mot correct
                        if correct_answer[0].isupper():
                            translation = translation.capitalize()
                        else:
                            translation = translation.lower()
                        distractors.append(translation)
                        
                except Exception as e:
                    print(f"Erreur traduction: {e}")
                    continue
        
        # Stratégie 2: Variations morphologiques si pas assez de distracteurs
        if len(distractors) < count:
            patterns = morphological_patterns.get(target_language, [])
            base = correct_answer.lower()
            
            for pattern_func, pattern_type in patterns:
                if len(distractors) >= count:
                    break
                
                try:
                    variant = pattern_func(base)
                    if is_valid_distractor(variant):
                        if correct_answer[0].isupper():
                            variant = variant.capitalize()
                        distractors.append(variant)
                except Exception:
                    continue
        
        # Stratégie 3: Génération par rhyme/assonance si encore insuffisant
        if len(distractors) < count:
            base = correct_answer.lower()
            if len(base) >= 3:
                letters = 'abcdefghijklmnopqrstuvwxyz'
                if target_language == 'de':
                    letters += 'äöüß'
                elif target_language == 'fr':
                    letters += 'éèêëàâäôùûüÿç'
                
                # Changer la première ou dernière lettre
                for letter in letters:
                    if len(distractors) >= count:
                        break
                        
                    variants = [
                        letter + base[1:],  # Première lettre
                        base[:-1] + letter  # Dernière lettre
                    ]
                    
                    for variant in variants:
                        if is_valid_distractor(variant):
                            if correct_answer[0].isupper():
                                variant = variant.capitalize()
                            distractors.append(variant)
                            if len(distractors) >= count:
                                break
        
        return distractors[:count]
    def _create_fallback_exercise(self, word: str, correct_answer: str, question_de: str, 
                                question_fr: str, question_type: str) -> Dict:
        """Crée un exercice de fallback avec des options prédéfinies"""
        
        if question_type == f"{self.source_lang}_to_{self.target_lang}":
            # Distracteurs pour la langue cible (ex: français)
            wrong_options = [
                correct_answer + "er",
                "le " + correct_answer, 
                correct_answer + "tion"
            ]
        else:
            # Distracteurs pour la langue source
            wrong_options = [
                correct_answer + "en",
                correct_answer + "er",
                "un" + correct_answer
            ]
        
        options = [correct_answer] + wrong_options
        random.shuffle(options)
        
        return {
            'type': 'multiple_choice',
            'question_type': question_type,
            'question': {
                self.source_lang: f"Mot dans {LANGUAGE_NAMES[self.target_lang]} pour '{word}'?",
                self.target_lang: f"Mot en {LANGUAGE_NAMES[self.source_lang]} pour '{word}'?"
            },
            'word_to_translate': word,
            'options': options,
            'correct_answer': correct_answer,
            'level': self.determine_word_level(word if question_type == f"{self.source_lang}_to_{self.target_lang}" else correct_answer),
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
        # Structure de dossiers simplifiée
        lang_dir = self.output_dir
        
        # Traductions et exercices
        translation_info = self.get_basic_translation(word)
        example = self.generate_simple_example(word)
        exercise = self.create_simple_exercise(
            word, 
            translation_info[self.target_lang]
        )
        level = self.determine_word_level(word)
        
        # Chercher et télécharger les prononciations
        audio_sources = self.scrape_audio_tags(word)
        pronunciations = []
        
        for idx, source in enumerate(audio_sources[:CONFIG['max_examples_per_word']]):
            filename = f"{word}_{idx + 1}{Path(source['url']).suffix}"
            filepath = lang_dir / f"pron_{filename}"
            
            if self.download_audio_file(source['url'], str(filepath)):
                pronunciations.append({
                    'file': str(filepath),
                    'type': source['type'],
                    'language': source['language']
                })
        
        # Créer le résultat JSON avec métadonnées de langue
        result = {
            'word': word,
            'timestamp': self.get_timestamp(),
            'translations': translation_info,
            'examples': example,
            'exercises': exercise,
            'level': level,
            'pronunciations': pronunciations,
            'metadata': {
                'source_language': self.source_lang,
                'target_language': self.target_lang,
                'source_lang_name': LANGUAGE_NAMES.get(self.source_lang, self.source_lang),
                'target_lang_name': LANGUAGE_NAMES.get(self.target_lang, self.target_lang)
            }
        }
        
        # Sauvegarder dans un seul fichier JSON
        word_file = lang_dir / f"{word}.json"
        with open(word_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        return result
    def get_timestamp(self) -> str:
        """Retourne un timestamp pour le traitement"""
        from datetime import datetime
        return datetime.now().isoformat()
class AudioTranscriber:
    """Classe pour gérer la transcription audio"""
    
    def __init__(self, source_lang: str = "de"):
        self.model = CONFIG['whisper_model']
        self.language = source_lang
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
    
    def load_models(self, lang_pair: str = "de-fr") -> Dict:
        """Charge les modèles pour une paire de langues donnée"""
        if lang_pair not in CONFIG['language_pairs']:
            raise ValueError(f"Paire de langues non supportée: {lang_pair}")
            
        lang_config = CONFIG['language_pairs'][lang_pair]
        source_lang, target_lang = lang_pair.split('-')
        
        try:
            # Modèle de traduction
            tokenizer = MarianTokenizer.from_pretrained(lang_config['translation_model_path'])
            model = MarianMTModel.from_pretrained(lang_config['translation_model_path'])
            
            # Modèles GPT source et cible
            tokenizer_gpt_source = AutoTokenizer.from_pretrained(lang_config['source_gpt_model'])
            model_gpt_source = GPT2LMHeadModel.from_pretrained(lang_config['source_gpt_model'])
            tokenizer_gpt_target = AutoTokenizer.from_pretrained(lang_config['target_gpt_model'])
            model_gpt_target = GPT2LMHeadModel.from_pretrained(lang_config['target_gpt_model'])
            
            # Configuration des pad tokens
            if tokenizer_gpt_source.pad_token is None:
                tokenizer_gpt_source.pad_token = tokenizer_gpt_source.eos_token
            if tokenizer_gpt_target.pad_token is None:
                tokenizer_gpt_target.pad_token = tokenizer_gpt_target.eos_token
            
            self.models = {
                lang_pair: (tokenizer, model),
                f'gpt_{source_lang}': (tokenizer_gpt_source, model_gpt_source),
                f'gpt_{target_lang}': (tokenizer_gpt_target, model_gpt_target)
            }
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
            output_dir = str(Path(json_path).parent)
            self.vocabulary_processor = VocabularyProcessor(models, output_dir)
            
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
    
    def process_audio_file(self, audio_path: str, output_dir: str = None) -> bool:
        """Traite un fichier audio complet"""
        # Validation
        if not self.validate_audio_file(audio_path):
            return False
        
        # Créer le dossier de sortie dans ~/sprech-audio/nom_audio/
        output_dir = str(get_output_folder(audio_path))
        print(f"Dossier de sortie: {output_dir}")
        
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
        description="Transcrit et analyse un fichier audio dans la langue source",
        epilog="Exemple: python script.py audio.mp3 --source-lang de --target-lang fr"
    )
    
    parser.add_argument(
        "audio_file",
        help="Fichier audio à traiter"
    )
    
    parser.add_argument(
        "--source-lang",
        default="de",
        help=f"Langue source (défaut: de). Langues supportées: {', '.join(LANGUAGE_NAMES.keys())}"
    )
    
    parser.add_argument(
        "--target-lang",
        default="fr",
        help=f"Langue cible (défaut: fr). Langues supportées: {', '.join(LANGUAGE_NAMES.keys())}"
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
    
    # Vérifier si la paire de langues est supportée
    lang_pair = f"{args.source_lang}-{args.target_lang}"
    if lang_pair not in CONFIG['language_pairs']:
        print(f"✗ Paire de langues non supportée: {lang_pair}")
        print(f"Paires supportées: {', '.join(CONFIG['language_pairs'].keys())}")
        sys.exit(1)
    
    # Mettre à jour la configuration
    CONFIG['max_vocabulary_size'] = args.max_vocab
    CONFIG['whisper_model'] = args.model
    CONFIG['language'] = args.source_lang
    
    # Traitement 
    processor = TranscriptionProcessor()
    processor.transcriber = AudioTranscriber(source_lang=args.source_lang)
    success = processor.process_audio_file(args.audio_file)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()