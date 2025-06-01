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
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, AutoTokenizer

from tqdm import tqdm


# Configuration globale
CONFIG = {
    "whisper_model": "base",
    "language": "de",
    "output_format": "json",
    "translation_model_path": "./opus-mt-de-fr",
    "german_gpt_model": "benjamin/gpt2-wechsel-german",
    "french_gpt_model": "dbddv01/gpt2-french-small",
    "max_vocabulary_size": 100,  # Limiter le nombre de mots à traiter
    "min_word_length": 2,
    "max_examples_per_word": 3,
    "max_exercises_per_type": 2
}

AUDIO_EXTENSIONS = (
    ".opus",
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".aac",
    ".aiff",
    ".wma",
)


def download_audio_file(audio_url, filename=None):
    """
    Télécharge un fichier audio
    """
    if not filename:
        filename = audio_url.split("/")[-1]
        if not any(ext in filename.lower() for ext in [".mp3", ".wav", ".ogg", ".m4a"]):
            filename += ".mp3"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(audio_url, headers=headers, stream=True)
        response.raise_for_status()

        with open(filename, "wb") as f:
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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        print(f"Recherche de prononciation pour : {word}")

        try:
            # Ajout d'un délai aléatoire entre 1 et 3 secondes
            time.sleep(random.uniform(1, 3))

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parser le HTML
            soup = BeautifulSoup(response.content, "html.parser")

            audio_sources = []

            # 1. Rechercher les balises audio directes
            audio_tags = soup.find_all("audio")
            for audio in audio_tags:
                if audio.get("src"):
                    src = audio.get("src")
                    full_url = urljoin(url, src)
                    audio_sources.append(
                        {"type": "direct_src", "url": full_url, "original_src": src}
                    )

                for source in audio.find_all("source"):
                    if source.get("src"):
                        src = source.get("src")
                        full_url = urljoin(url, src)
                        audio_sources.append(
                            {
                                "type": "source_tag",
                                "url": full_url,
                                "original_src": src,
                                "mime_type": source.get("type", "unknown"),
                            }
                        )

            # 2. Rechercher dans le JavaScript
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    # Chercher des URLs audio
                    audio_urls = re.findall(
                        r'["\']([^"\']*\.(?:mp3|wav|ogg|m4a)[^"\']*)["\']',
                        script.string,
                    )
                    for audio_url in audio_urls:
                        full_url = urljoin(url, audio_url)
                        audio_sources.append(
                            {
                                "type": "js_embedded",
                                "url": full_url,
                                "original_src": audio_url,
                            }
                        )

            # 3. Rechercher les attributs data-*
            elements_with_data = soup.find_all(
                lambda tag: any(attr.startswith("data-") for attr in tag.attrs)
            )
            for element in elements_with_data:
                for attr, value in element.attrs.items():
                    if attr.startswith("data-") and isinstance(value, str):
                        if any(
                            ext in value.lower()
                            for ext in [".mp3", ".wav", ".ogg", ".m4a"]
                        ):
                            full_url = urljoin(url, value)
                            audio_sources.append(
                                {
                                    "type": f"data_attribute_{attr}",
                                    "url": full_url,
                                    "original_src": value,
                                }
                            )

            print(f"✓ {len(audio_sources)} sources audio trouvées pour '{word}'")
            return audio_sources

        except Exception as e:
            print(f"✗ Erreur lors du scraping pour '{word}': {e}")
            return []

    def download_audio_file(self, audio_url: str, filename: str) -> bool:
        """Télécharge un fichier audio de prononciation"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            response = requests.get(audio_url, headers=headers, stream=True)
            response.raise_for_status()

            filepath = self.pronunciation_dir / filename
            with open(filepath, "wb") as f:
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
        word = re.sub(r"^[^\w\säöüß]*|[^\w\säöüß]*$", "", word)

        # Valider le mot
        if (
            len(word) < CONFIG["min_word_length"]
            or len(word) > 20
            or not re.match(r"^[a-zA-ZäöüßÄÖÜ]+$", word)
            or word in self.processed_words
        ):
            return None

        self.processed_words.add(word)
        return word

    def extract_vocabulary(self, segments: List[Dict]) -> List[str]:
        """Extrait et nettoie le vocabulaire des segments transcrits"""
        vocabulary = set()

        for segment in segments:
            text = segment.get("text", "").strip()
            words = text.split()

            for word in words:
                clean_word = self.clean_word(word)
                if clean_word and len(vocabulary) < CONFIG["max_vocabulary_size"]:
                    vocabulary.add(clean_word)

        return sorted(list(vocabulary))

    def get_basic_translation(self, word: str) -> Dict[str, str]:
        """Obtient la traduction de base d'un mot"""
        tokenizer_de_fr, model_de_fr = self.models["de_fr"]

        try:
            inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=20,
                num_beams=3,
                temperature=0.3,
                do_sample=False,
            )
            translation = tokenizer_de_fr.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            return {"de": word, "fr": translation if translation else word}
        except Exception as e:
            print(f"Erreur de traduction pour '{word}': {e}")
            return {"de": word, "fr": word}

    def generate_simple_example(self, word: str) -> Dict[str, str]:
        """Génère un exemple simple d'utilisation"""
        tokenizer_gpt, model_gpt = self.models["gpt"]
        tokenizer_de_fr, model_de_fr = self.models["de_fr"]

        # Templates simples prédéfinis
        simple_templates = [
            f"Das ist ein {word}.",
            f"Ich habe einen {word}.",
            f"Der {word} ist schön.",
            f"Wir brauchen {word}.",
        ]

        try:
            # Choisir un template au hasard
            german_sentence = random.choice(simple_templates)

            # Traduire en français
            inputs = tokenizer_de_fr(german_sentence, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids, max_length=30, num_beams=3, temperature=0.3
            )
            french_sentence = tokenizer_de_fr.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            return {
                "de": german_sentence,
                "fr": french_sentence if french_sentence else "Phrase d'exemple",
            }

        except Exception as e:
            print(f"Erreur génération exemple pour '{word}': {e}")
            return {"de": f"Das ist {word}.", "fr": f"C'est {word}."}

    def create_simple_exercise(self, word: str, translation: str) -> Dict:
        """Crée deux exercices à choix multiples (QCM) dans les deux sens de traduction"""

        # 1. Exercice DE -> FR
        de_fr_exercise = self._create_exercise_variant(
            word=word,
            correct_answer=translation,
            question_type="de_to_fr",
            question_de=f"Welches französische Wort bedeutet '{word}'?",
            question_fr=f"Quelle est la traduction française de '{word}'?",
            distractor_language="fr",
        )

        # 2. Exercice FR -> DE
        fr_de_exercise = self._create_exercise_variant(
            word=translation,
            correct_answer=word,
            question_type="fr_to_de",
            question_de=f"Welches deutsche Wort bedeutet '{translation}'?",
            question_fr=f"Quel est le mot allemand pour '{translation}'?",
            distractor_language="de",
        )

        return {
            "type": "multiple_choice_pair",
            "de_to_fr": de_fr_exercise,
            "fr_to_de": fr_de_exercise,
            "level": self.determine_word_level(word),
        }

    def _create_exercise_variant(
        self,
        word: str,
        correct_answer: str,
        question_type: str,
        question_de: str,
        question_fr: str,
        distractor_language: str,
    ) -> Dict:
        """Crée un exercice QCM dans un sens de traduction spécifique"""

        try:
            # Générer les distracteurs (mauvaises réponses)
            wrong_options = self._generate_distractors(
                word=word,
                correct_answer=correct_answer,
                target_language=distractor_language,
                count=3,
            )

            # Créer la liste des options avec la bonne réponse
            options = [correct_answer] + wrong_options
            random.shuffle(options)

            return {
                "type": "multiple_choice",
                "question_type": question_type,
                "question": {"de": question_de, "fr": question_fr},
                "word_to_translate": word,
                "options": options,
                "correct_answer": correct_answer,
                "level": self.determine_word_level(
                    word if question_type == "de_to_fr" else correct_answer
                ),
            }

        except Exception as e:
            print(f"Erreur lors de la génération de l'exercice pour '{word}': {e}")
            # Fallback avec des options simples
            return self._create_fallback_exercise(
                word, correct_answer, question_de, question_fr, question_type
            )

    def _generate_distractors(
        self, word: str, correct_answer: str, target_language: str, count: int = 3
    ) -> List[str]:
        """Génère des distracteurs (mauvaises réponses) pour le QCM"""
        distractors = []
        
        if target_language == "fr":
            # Utiliser le modèle GPT français pour générer des mots français
            distractors = self._generate_french_words(correct_answer, count)
        else:
            # Utiliser le modèle GPT allemand pour générer des mots allemands
            distractors = self._generate_german_words(correct_answer, count)

        return distractors[:count]
    def _generate_french_words(self, correct_answer: str, count: int) -> List[str]:
        """Génère des mots français avec le modèle GPT français"""
        tokenizer_gpt_fr, model_gpt_fr = self.models["gpt_fr"]
        distractors = []
        
        # Déterminer la catégorie du mot pour créer des prompts appropriés
        category = self._guess_word_category(correct_answer)
        
        # Créer des prompts contextuels pour générer des mots de la même catégorie
        prompts = self._create_french_prompts(correct_answer, category)
        
        for prompt in prompts:
            if len(distractors) >= count:
                break
                
            try:
                # Tokeniser le prompt
                inputs = tokenizer_gpt_fr(prompt, return_tensors="pt", padding=True)
                
                # Générer du texte
                with torch.no_grad():
                    outputs = model_gpt_fr.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 10,
                        num_return_sequences=2,  # Générer plusieurs variantes
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer_gpt_fr.eos_token_id,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=2
                    )
                
                # Extraire les mots générés
                for output in outputs:
                    generated_text = tokenizer_gpt_fr.decode(output, skip_special_tokens=True)
                    new_words = self._extract_words_from_generation(generated_text, prompt, "fr")
                    
                    for new_word in new_words:
                        if (new_word != correct_answer.lower() and 
                            new_word not in [d.lower() for d in distractors] and
                            len(distractors) < count and
                            self._is_valid_french_word(new_word)):
                            
                            # Maintenir la casse appropriée
                            formatted_word = self._format_word_case(new_word, correct_answer)
                            distractors.append(formatted_word)
                            
            except Exception as e:
                print(f"Erreur génération mot français: {e}")
                continue
        
        # Si pas assez de mots générés, utiliser des variations morphologiques
        while len(distractors) < count:
            variations = self._create_morphological_variations(correct_answer, "fr")
            for var in variations:
                if (var not in distractors and 
                    var != correct_answer and 
                    len(distractors) < count):
                    distractors.append(var)
        
        return distractors

    def _generate_german_words(self, correct_answer: str, count: int) -> List[str]:
        """Génère des mots allemands avec le modèle GPT allemand"""
        tokenizer_gpt_de, model_gpt_de = self.models["gpt_de"]
        distractors = []
        
        # Créer des prompts pour générer des mots allemands
        prompts = self._create_german_prompts(correct_answer)
        
        for prompt in prompts:
            if len(distractors) >= count:
                break
                
            try:
                inputs = tokenizer_gpt_de(prompt, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    outputs = model_gpt_de.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 10,
                        num_return_sequences=2,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer_gpt_de.eos_token_id,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=2
                    )
                
                for output in outputs:
                    generated_text = tokenizer_gpt_de.decode(output, skip_special_tokens=True)
                    new_words = self._extract_words_from_generation(generated_text, prompt, "de")
                    
                    for new_word in new_words:
                        if (new_word != correct_answer.lower() and 
                            new_word not in [d.lower() for d in distractors] and
                            len(distractors) < count and
                            self._is_valid_german_word(new_word)):
                            
                            formatted_word = self._format_word_case(new_word, correct_answer)
                            distractors.append(formatted_word)
                            
            except Exception as e:
                print(f"Erreur génération mot allemand: {e}")
                continue
        
        # Compléter avec des variations si nécessaire
        while len(distractors) < count:
            variations = self._create_morphological_variations(correct_answer, "de")
            for var in variations:
                if (var not in distractors and 
                    var != correct_answer and 
                    len(distractors) < count):
                    distractors.append(var)
        
        return distractors

    def _create_french_prompts(self, word: str, category: str) -> List[str]:
        """Crée des prompts contextuels pour générer des mots français"""
        word_lower = word.lower()
        
        if category == "noun":
            return [
                f"Voici des noms français: maison, voiture, {word_lower},",
                f"Liste de substantifs: livre, table, {word_lower},",
                f"Des objets: fenêtre, porte, {word_lower},",
                f"Mots français: chat, chien, {word_lower},"
            ]
        elif category == "verb":
            return [
                f"Verbes français: faire, avoir, {word_lower},",
                f"Actions: aller, venir, {word_lower},",
                f"Des verbes: manger, boire, {word_lower},",
                f"Infinitifs: voir, savoir, {word_lower},"
            ]
        elif category == "adjective":
            return [
                f"Adjectifs français: grand, petit, {word_lower},",
                f"Qualités: beau, joli, {word_lower},",
                f"Descriptions: bon, mauvais, {word_lower},",
                f"Caractéristiques: nouveau, vieux, {word_lower},"
            ]
        else:
            return [
                f"Mots français courants: temps, jour, {word_lower},",
                f"Vocabulaire français: monde, vie, {word_lower},",
                f"Lexique: famille, ami, {word_lower},",
                f"Termes français: travail, école, {word_lower},"
            ]

    def _create_german_prompts(self, word: str) -> List[str]:
        """Crée des prompts pour générer des mots allemands"""
        word_lower = word.lower()
        
        return [
            f"Deutsche Wörter: Haus, Auto, {word_lower},",
            f"Substantive: Buch, Tisch, {word_lower},",
            f"Begriffe: Wasser, Brot, {word_lower},",
            f"Vokabular: Zeit, Leben, {word_lower},",
            f"Deutsche Begriffe: Mann, Frau, {word_lower},"
        ]

    def _extract_words_from_generation(self, generated_text: str, prompt: str, language: str) -> List[str]:
        """Extrait des mots valides du texte généré"""
        # Enlever le prompt du texte généré
        new_text = generated_text.replace(prompt, "").strip()
        
        # Extraire les mots
        if language == "fr":
            # Pour le français, extraire les mots avec accents
            words = re.findall(r'\b[a-zA-ZàâäçéèêëïîôöùûüÿÀÂÄÇÉÈÊËÏÎÔÖÙÛÜŸ]+\b', new_text)
        else:
            # Pour l'allemand, inclure les Umlauts
            words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]+\b', new_text)
        
        # Nettoyer et filtrer les mots
        valid_words = []
        for word in words:
            word = word.strip().lower()
            if (len(word) >= 3 and 
                len(word) <= 15 and 
                not word.isdigit() and
                word not in ['und', 'der', 'die', 'das', 'ein', 'eine', 'le', 'la', 'les', 'un', 'une', 'des']):
                valid_words.append(word)
        
        return valid_words[:5]  # Limiter à 5 mots par génération

    def _guess_word_category(self, word: str) -> str:
        """Devine la catégorie grammaticale d'un mot français"""
        word_lower = word.lower()
        
        # Terminaisons typiques des noms
        if any(word_lower.endswith(ending) for ending in 
               ['tion', 'sion', 'ment', 'ence', 'ance', 'eur', 'euse', 'teur', 'trice', 'age', 'ise']):
            return "noun"
        
        # Terminaisons typiques des verbes
        if any(word_lower.endswith(ending) for ending in ['er', 'ir', 're']):
            return "verb"
        
        # Terminaisons typiques des adjectifs
        if any(word_lower.endswith(ending) for ending in 
               ['able', 'ible', 'eux', 'euse', 'ant', 'ent', 'if', 'ive']):
            return "adjective"
        
        return "noun"  # Par défaut

    def _is_valid_french_word(self, word: str) -> bool:
        """Vérifie si un mot français est valide"""
        return (len(word) >= 3 and 
                len(word) <= 15 and 
                re.match(r'^[a-zA-ZàâäçéèêëïîôöùûüÿÀÂÄÇÉÈÊËÏÎÔÖÙÛÜŸ]+$', word) and
                word.lower() not in ['les', 'des', 'une', 'est', 'sont', 'ont', 'aux'])

    def _is_valid_german_word(self, word: str) -> bool:
        """Vérifie si un mot allemand est valide"""
        return (len(word) >= 3 and 
                len(word) <= 15 and 
                re.match(r'^[a-zA-ZäöüßÄÖÜ]+$', word) and
                word.lower() not in ['der', 'die', 'das', 'und', 'mit', 'von', 'für', 'auf'])

    def _format_word_case(self, word: str, reference_word: str) -> str:
        """Formate la casse d'un mot en se basant sur un mot de référence"""
        if reference_word[0].isupper():
            return word.capitalize()
        return word.lower()

    def _create_morphological_variations(self, word: str, language: str) -> List[str]:
        """Crée des variations morphologiques d'un mot"""
        variations = []
        word_lower = word.lower()
        
        if language == "fr":
            # Variations françaises
            suffixes = ['ment', 'tion', 'eur', 'euse', 'able', 'ique', 'isme']
            prefixes = ['pré', 'sur', 'sous', 'anti', 're']
        else:
            # Variations allemandes
            suffixes = ['ung', 'heit', 'keit', 'lich', 'isch', 'bar', 'los']
            prefixes = ['un', 'vor', 'nach', 'über', 'unter']
        
        # Ajouter des suffixes
        for suffix in suffixes[:3]:  # Limiter à 3 suffixes
            if not word_lower.endswith(suffix):
                if word_lower.endswith('e') and language == "fr":
                    variations.append(f"{word_lower[:-1]}{suffix}")
                else:
                    variations.append(f"{word_lower}{suffix}")
        
        # Ajouter des préfixes
        for prefix in prefixes[:2]:  # Limiter à 2 préfixes
            variations.append(f"{prefix}{word_lower}")
        
        return variations
    def _create_fallback_exercise(
        self,
        word: str,
        correct_answer: str,
        question_de: str,
        question_fr: str,
        question_type: str,
    ) -> Dict:
        """Crée un exercice de fallback avec des options prédéfinies"""

        if question_type == "de_to_fr":
            # Distracteurs français simples
            wrong_options = [
                correct_answer + "er",
                "un " + correct_answer,
                correct_answer + "tion",
            ]
        else:
            # Distracteurs allemands simples
            wrong_options = [
                correct_answer + "en",
                "der " + correct_answer,
                correct_answer + "ung",
            ]

        options = [correct_answer] + wrong_options
        random.shuffle(options)

        return {
            "type": "multiple_choice",
            "question_type": question_type,
            "question": {"de": question_de, "fr": question_fr},
            "word_to_translate": word,
            "options": options,
            "correct_answer": correct_answer,
            "level": self.determine_word_level(
                word if question_type == "de_to_fr" else correct_answer
            ),
            "fallback": True,
        }

    def determine_word_level(self, word: str) -> str:
        """Détermine le niveau de difficulté du mot"""
        if len(word) <= 4:
            return "beginner"
        elif len(word) <= 8:
            return "intermediate"
        else:
            return "advanced"

    def process_word(self, word: str) -> Dict:
        """Traite un mot complet avec traduction, exemple et exercice"""
        # Traductions et exercices
        translation_info = self.get_basic_translation(word)
        example = self.generate_simple_example(word)
        exercise = self.create_simple_exercise(word, translation_info["fr"])
        level = self.determine_word_level(word)

        # Récupération de la prononciation
        pronunciation_info = {"available": False, "file": None}
        audio_sources = self.scrape_audio_tags(word)

        if audio_sources:
            # Tenter de télécharger le premier fichier audio trouvé
            first_audio = audio_sources[0]
            audio_filename = f"{word}_pronunciation.mp3"

            if self.download_audio_file(first_audio["url"], audio_filename):
                pronunciation_info = {
                    "available": True,
                    "file": str(self.pronunciation_dir / audio_filename),
                }

        # Construire le résultat
        result = {
            "word": word,
            "level": level,
            "translation": translation_info,
            "example": example,
            "exercise": exercise,
            "pronunciation": pronunciation_info,
            "processed_at": self.get_timestamp(),
        }

        return result

    def get_timestamp(self) -> str:
        """Retourne un timestamp pour le traitement"""
        from datetime import datetime

        return datetime.now().isoformat()


class AudioTranscriber:
    """Classe pour gérer la transcription audio"""

    def __init__(self):
        self.model = CONFIG["whisper_model"]
        self.language = CONFIG["language"]
        self.output_format = CONFIG["output_format"]

    def transcribe_file(self, audio_path: str, output_directory: str) -> Optional[str]:
        """Transcrit un fichier audio"""
        print(f"Transcription de: {audio_path}")

        base_name = Path(audio_path).stem
        output_filename = f"{base_name}.{self.output_format}"
        output_file_path = Path(output_directory) / output_filename

        command = [
            sys.executable,
            "-m",
            "whisperx",
            audio_path,
            "--model",
            self.model,
            "--language",
            self.language,
            "--output_format",
            self.output_format,
            "--output_dir",
            output_directory,
            "--segment_resolution",
            "chunk",
            "--max_line_count",
            "1",
            "--compute_type",
            "float32",
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
            tokenizer_de_fr = MarianTokenizer.from_pretrained(
                CONFIG["translation_model_path"]
            )
            model_de_fr = MarianMTModel.from_pretrained(
                CONFIG["translation_model_path"]
            )

            # Modèle GPT allemand
            print("→ Chargement du modèle GPT allemand...")
            tokenizer_gpt_de = AutoTokenizer.from_pretrained(CONFIG["german_gpt_model"])
            model_gpt_de = GPT2LMHeadModel.from_pretrained(CONFIG["german_gpt_model"])
            if tokenizer_gpt_de.pad_token is None:
                tokenizer_gpt_de.pad_token = tokenizer_gpt_de.eos_token


            print("→ Chargement du modèle GPT français...")
            tokenizer_gpt_fr = AutoTokenizer.from_pretrained(CONFIG["french_gpt_model"])
            model_gpt_fr = GPT2LMHeadModel.from_pretrained(CONFIG["french_gpt_model"])
            if tokenizer_gpt_fr.pad_token is None:
                tokenizer_gpt_fr.pad_token = tokenizer_gpt_fr.eos_token


           
            self.models = {
                "de_fr": (tokenizer_de_fr, model_de_fr),
                "gpt_de": (tokenizer_gpt_de, model_gpt_de),
                "gpt_fr": (tokenizer_gpt_fr, model_gpt_fr),
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
            print(
                f"✗ Format audio non supporté. Formats acceptés: {', '.join(AUDIO_EXTENSIONS)}"
            )
            return False

        return True

    def process_transcription_file(self, json_path: str) -> bool:
        """Traite le fichier JSON de transcription"""
        try:
            # Charger le fichier JSON
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "segments" not in data:
                print("✗ Format JSON invalide - 'segments' manquant")
                return False

            # Initialiser le processeur de vocabulaire
            models = self.model_manager.load_models()
            self.vocabulary_processor = VocabularyProcessor(models)

            # Traiter chaque segment et ajouter la traduction
            print("Traduction des segments...")
            for i, segment in enumerate(tqdm(data["segments"], desc="Segments")):
                text = segment.get("text", "").strip()
                if text:
                    translation_info = self.vocabulary_processor.get_basic_translation(
                        text
                    )
                    segment["translation"] = translation_info["fr"]

            # Extraire et traiter le vocabulaire
            print("Extraction du vocabulaire...")
            vocabulary_words = self.vocabulary_processor.extract_vocabulary(
                data["segments"]
            )

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
            data["vocabulary"] = vocabulary_entries
            data["vocabulary_stats"] = {
                "total_words": len(vocabulary_entries),
                "levels": {
                    "beginner": len(
                        [w for w in vocabulary_entries if w["level"] == "beginner"]
                    ),
                    "intermediate": len(
                        [w for w in vocabulary_entries if w["level"] == "intermediate"]
                    ),
                    "advanced": len(
                        [w for w in vocabulary_entries if w["level"] == "advanced"]
                    ),
                },
            }

            # Sauvegarder le fichier modifié
            with open(json_path, "w", encoding="utf-8") as f:
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
        epilog="Exemple: python script.py audio.mp3 -o ./output",
    )

    parser.add_argument("audio_file", help="Fichier audio à traiter")

    parser.add_argument(
        "-o", "--output", help="Répertoire de sortie (obligatoire)", required=True
    )

    parser.add_argument(
        "--max-vocab",
        type=int,
        default=CONFIG["max_vocabulary_size"],
        help=f"Nombre maximum de mots à traiter (défaut: {CONFIG['max_vocabulary_size']})",
    )

    parser.add_argument(
        "--model",
        default=CONFIG["whisper_model"],
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Modèle Whisper à utiliser (défaut: {CONFIG['whisper_model']})",
    )

    args = parser.parse_args()

    # Mettre à jour la configuration
    CONFIG["max_vocabulary_size"] = args.max_vocab
    CONFIG["whisper_model"] = args.model

    # Traitement
    processor = TranscriptionProcessor()
    success = processor.process_audio_file(args.audio_file, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
