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
    'max_exercises_per_word': 3,  # Augmenté pour plus d'exercices
    'definition_templates': [
        "Was ist {}?",
        "Erklären Sie {}:",
        "Definition von {}:",
        "{} bedeutet:",
        "Beschreiben Sie {}:"
    ]
}

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

# Types d'exercices disponibles
EXERCISE_TYPES = [
    'fill_blank',
    'multiple_choice',
    'translation',
    'sentence_building',
    'synonym_antonym',
    'definition_match',
    'conjugation',
    'gender_article'
]

class VocabularyProcessor:
    """Classe pour gérer le traitement du vocabulaire et les analyses lexicales"""
    
    def __init__(self, models: Dict):
        self.models = models
        self.processed_words = set()
        self.common_articles = ['der', 'die', 'das', 'ein', 'eine', 'einen', 'einem', 'einer']
        self.common_distractors = ['haus', 'auto', 'buch', 'tisch', 'stuhl', 'wasser', 'brot', 'milch']
        
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
    
    def generate_definition(self, word: str) -> Dict[str, str]:
        """Génère une définition du mot en allemand et en français"""
        tokenizer_gpt, model_gpt = self.models['gpt']
        tokenizer_de_fr, model_de_fr = self.models['de_fr']
        
        # Templates de définition plus spécifiques selon le type de mot
        definition_templates = {
            'verbs': [
                "Das Verb '{}' bedeutet une action, bei der man",
                "Mit dem Verb '{}' beschreibt man une activité, wobei",
                "Wenn man '{}' tut, dann"
            ],
            'nouns': [
                "Ein {} ist ein Gegenstand, der",
                "Das Wort {} bezeichnet",
                "Als {} versteht man"
            ],
            'time_words': [
                "{} ist ein Zeitbegriff, der",
                "Der Ausdruck {} bezieht sich auf",
                "Mit {} meint man den Zeitpunkt"
            ],
            'adjectives': [
                "Das Adjektiv {} beschreibt",
                "Mit {} bezeichnet man",
                "Etwas, das {} ist, hat die Eigenschaft"
            ],
            'prepositions': [
                "Die Präposition {} wird verwendet, wenn",
                "{} zeigt eine Beziehung zwischen",
                "Man benutzt {}, um auszudrücken"
            ],
            'general': [
                "{} ist ein wichtiges Wort, das",
                "Der Begriff {} wird verwendet für",
                "Mit {} bezeichnet man"
            ]
        }

        try:
            # Déterminer le type de mot
            if word.endswith('en'):
                word_type = 'verbs'
            elif word in ['heute', 'morgen', 'jetzt', 'später']:
                word_type = 'time_words'
            elif word.startswith(('der', 'die', 'das', 'ein', 'eine')):
                word_type = 'nouns'
            elif len(word) <= 4 and word in ['in', 'auf', 'mit', 'bei', 'zu', 'aus', 'von', 'nach', 'ins']:
                word_type = 'prepositions'
            else:
                word_type = 'general'

            # Sélectionner un template approprié
            templates = definition_templates.get(word_type, definition_templates['general'])
            definition_prompt = random.choice(templates).format(word)
            
            # Générer la définition en allemand
            inputs = tokenizer_gpt(definition_prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=5,  # Augmenté pour plus de cohérence
                temperature=0.7,
                do_sample=True,
                no_repeat_ngram_size=3,  # Éviter les répétitions de phrases
                pad_token_id=tokenizer_gpt.eos_token_id
            )
            
            german_definition = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
            german_definition = german_definition.replace(definition_prompt, "").strip()
            
            # Validation de la définition
            if not german_definition or len(german_definition.split()) < 5:
                fallback_definitions = {
                    'verbs': f"Das Verb '{word}' beschreibt une importante Handlung im Alltag.",
                    'nouns': f"Ein {word} ist ein wichtiger Gegenstand im täglichen Leben.",
                    'time_words': f"{word} bezeichnet einen bestimmten Zeitpunkt oder Zeitraum.",
                    'prepositions': f"Die Präposition {word} zeigt une räumliche oder zeitliche Beziehung.",
                    'general': f"{word} ist ein nützliches deutsches Wort im Alltag."
                }
                german_definition = fallback_definitions.get(word_type, fallback_definitions['general'])
            
            # Traduire la définition en français
            inputs_fr = tokenizer_de_fr(german_definition, return_tensors="pt", padding=True)
            outputs_fr = model_de_fr.generate(
                inputs_fr.input_ids,
                max_length=60,
                num_beams=5,
                temperature=0.3,
                no_repeat_ngram_size=3
            )
            french_definition = tokenizer_de_fr.decode(outputs_fr[0], skip_special_tokens=True).strip()
            
            return {
                'de': german_definition,
                'fr': french_definition if french_definition else f"Définition de {word}"
            }
            
        except Exception as e:
            print(f"Erreur génération définition pour '{word}': {e}")
            fallback = {
                'verbs': (f"Das Verb '{word}' beschreibt une Aktion.", f"Le verbe '{word}' décrit une action."),
                'nouns': (f"'{word}' ist ein nützliches Substantiv.", f"'{word}' est un nom utile."),
                'time_words': (f"'{word}' bezieht sich auf die Zeit.", f"'{word}' fait référence au temps."),
                'prepositions': (f"'{word}' ist eine wichtige Präposition.", f"'{word}' est une préposition importante."),
                'general': (f"'{word}' ist ein deutsches Wort.", f"'{word}' est un mot allemand.")
            }
            de_fallback, fr_fallback = fallback.get(word_type, fallback['general'])
            return {
                'de': de_fallback,
                'fr': fr_fallback
            }
    
    def generate_simple_example(self, word: str) -> Dict[str, str]:
        """Génère un exemple de phrase naturelle en utilisant le modèle GPT"""
        tokenizer_de_fr, model_de_fr = self.models['de_fr']
        tokenizer_gpt, model_gpt = self.models['gpt']
        
        try:
            # Déterminer le type de mot
            word_type = 'general'
            if word.endswith('en'):
                word_type = 'verbs'
            elif word in ['heute', 'morgen', 'jetzt', 'später']:
                word_type = 'time_words'
            elif word.startswith(('der', 'die', 'das', 'ein', 'eine')):
                word_type = 'nouns'
            elif len(word) <= 4 and word in ['in', 'auf', 'mit', 'bei', 'zu', 'aus', 'von', 'nach', 'ins']:
                word_type = 'prepositions'
            elif word in ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'Sie']:
                word_type = 'pronouns'

            # Créer un prompt contextuel selon le type de mot
            prompts = {
                'verbs': [
                    f"Schreiben Sie einen einfachen Satz mit dem Verb '{word}' im Präsens.",
                    f"Beschreiben Sie mit dem Verb '{word}' eine alltägliche Situation.",
                    f"Verwenden Sie das Verb '{word}' in einem natürlichen Kontext."
                ],
                'nouns': [
                    f"Beschreiben Sie kurz, wo sich {word} befindet.",
                    f"Was macht man mit {word}? Antworten Sie in einem Satz.",
                    f"Verwenden Sie {word} in einem einfachen Beispielsatz."
                ],
                'time_words': [
                    f"Wann passiert etwas? Verwenden Sie {word} in Ihrer Antwort.",
                    f"Beschreiben Sie mit {word} einen Zeitpunkt in Ihrem Tag.",
                    f"Wann machen Sie etwas? Nutzen Sie {word} in der Antwort."
                ],
                'prepositions': [
                    f"Wo liegt etwas? Nutzen Sie die Präposition '{word}'.",
                    f"Beschreiben Sie eine räumliche Beziehung mit '{word}'.",
                    f"Verwenden Sie '{word}' um zu zeigen, wo sich etwas befindet."
                ],
                'pronouns': [
                    f"Was macht {word}? Beschreiben Sie eine einfache Aktivität.",
                    f"Erzählen Sie kurz, was {word} heute macht.",
                    f"Verwenden Sie {word} in einem Satz über den Alltag."
                ],
                'general': [
                    f"Bilden Sie einen einfachen Satz mit '{word}'.",
                    f"Wie können Sie '{word}' in einem klaren Kontext verwenden?",
                    f"Schreiben Sie einen natürlichen Satz mit '{word}'."
                ]
            }
            # Choisir un prompt aléatoire pour plus de variété
            prompt = random.choice(prompts[word_type])

            # Générer une phrase en utilisant le modèle GPT
            prompt = prompts[word_type]
            input_ids = tokenizer_gpt(prompt, return_tensors="pt").input_ids
            outputs = model_gpt.generate(
                input_ids,
                max_length=50,
                num_return_sequences=3,  # Générer plusieurs phrases pour choisir la meilleure
                temperature=0.8,  # Légèrement plus créatif
                top_p=0.9,
                do_sample=True,
                no_repeat_ngram_size=3,  # Éviter les répétitions
                num_beams=5,  # Recherche de faisceau pour de meilleurs résultats
                early_stopping=True
            )
            # Décodage et sélection de la meilleure phrase
            examples = [tokenizer_gpt.decode(output, skip_special_tokens=True).strip()
                      for output in outputs]
            
            # Filtrer et évaluer les phrases générées
            valid_examples = []
            for ex in examples:
                # Vérifier les critères de base
                if not word.lower() in ex.lower() or len(ex.split()) < 4:
                    continue

                # Calcul du score de qualité
                score = 0
                
                # Longueur optimale (entre 5 et 12 mots)
                words = len(ex.split())
                if 5 <= words <= 12:
                    score += 3
                elif words < 15:
                    score += 1
                
                # Présence de ponctuation appropriée
                if ex.endswith(('.', '!', '?')):
                    score += 2
                
                # Vérification de la structure selon le type de mot
                if word_type == 'verbs':
                    # Le verbe doit être conjugué
                    if any(ex.lower().startswith(p.lower()) for p in ['ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr']):
                        score += 3
                elif word_type == 'nouns':
                    # Le nom doit avoir un article
                    if any(article.lower() + ' ' + word.lower() in ex.lower() 
                          for article in ['der', 'die', 'das', 'ein', 'eine']):
                        score += 3
                elif word_type == 'prepositions':
                    # La préposition doit être suivie d'un groupe nominal
                    if word.lower() + ' ' in ex.lower():
                        score += 3
                
                # Bonus pour les expressions courantes
                common_phrases = ['ist', 'hat', 'kann', 'möchte', 'muss', 'werden', 'haben']
                if any(phrase in ex.lower() for phrase in common_phrases):
                    score += 1
                
                valid_examples.append((ex, score))
            
            # Choisir la meilleure phrase ou utiliser un fallback
            if valid_examples:
                example = max(valid_examples, key=lambda x: x[1])[0]
            else:
                # Fallback plus naturel selon le type de mot
                fallbacks = {
                    'verbs': f"Ich {word} jeden Tag.",
                    'nouns': f"Das {word} ist sehr interessant.",
                    'time_words': f"{word} gehen wir spazieren.",
                    'prepositions': f"Das Buch liegt {word} dem Tisch.",
                    'pronouns': f"{word} geht zur Schule.",
                    'general': f"Das Wort '{word}' ist sehr wichtig."
                }
                example = fallbacks.get(word_type, fallbacks['general'])
            
            # Traduire l'exemple en français
            input_ids = tokenizer_de_fr(example, return_tensors="pt").input_ids
            outputs = model_de_fr.generate(input_ids, max_length=100)
            translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'de': example,
                'fr': translation
            }
            
        except Exception as e:
            print(f"Erreur génération exemple pour '{word}': {e}")
            # Fallback avec une phrase simple mais correcte
            return {
                'de': f"Das Wort '{word}' ist sehr nützlich.",
                'fr': f"Le mot '{word}' est très utile."
            }
    
    def create_fill_blank_exercise(self, word: str, translation: str) -> Dict:
        """Crée un exercice à trous avec plusieurs phrases d'exemple

        Le mot manquant peut être placé au début, au milieu ou à la fin de la phrase
        selon le template choisi. Chaque exercice contient plusieurs phrases d'exemple.
        """
        # Templates plus naturels et adaptés au contexte avec leurs traductions
        templates = {
            'verbs': {
                'middle': [
                    {
                        'de': "Die Kinder ____ durch den Garten.",
                        'de_solution': "Die Kinder gehen durch den Garten.",
                        'fr_solution': "Les enfants marchent à travers le jardin."
                    },
                    {
                        'de': "Wir ____ jeden Tag spazieren.",
                        'de_solution': "Wir gehen jeden Tag spazieren.",
                        'fr_solution': "Nous allons nous promener tous les jours."
                    }
                ],
                'end': [
                    {
                        'de': "Wir wollen zum Park ____.",
                        'de_solution': "Wir wollen zum Park gehen.",
                        'fr_solution': "Nous voulons aller au parc."
                    },
                    {
                        'de': "Sie muss zur Arbeit ____.",
                        'de_solution': "Sie muss zur Arbeit gehen.",
                        'fr_solution': "Elle doit aller au travail."
                    }
                ]
            },
            'nouns': {
                'middle': [
                    {
                        'de': "Der ____ steht im Wohnzimmer.",
                        'de_solution': "Der {} steht im Wohnzimmer.",
                        'fr_solution': "Le {} est dans le salon."
                    },
                    {
                        'de': "Eine ____ liegt auf dem Tisch.",
                        'de_solution': "Eine {} liegt auf dem Tisch.",
                        'fr_solution': "Un {} est sur la table."
                    }
                ],
                'end': [
                    {
                        'de': "Ich suche meinen ____.",
                        'de_solution': "Ich suche meinen {}.",
                        'fr_solution': "Je cherche mon {}."
                    },
                    {
                        'de': "Wir brauchen einen ____.",
                        'de_solution': "Wir brauchen einen {}.",
                        'fr_solution': "Nous avons besoin d'un {}."
                    }
                ]
            },
            'prepositions': {
                'middle': [
                    {
                        'de': "Das Buch liegt ____ dem Tisch.",
                        'de_solution': "Das Buch liegt {} dem Tisch.",
                        'fr_solution': "Le livre est {} la table."
                    },
                    {
                        'de': "Wir gehen ____ den Park.",
                        'de_solution': "Wir gehen {} den Park.",
                        'fr_solution': "Nous allons {} le parc."
                    }
                ]
            },
            'general': {
                'middle': [
                    {
                        'de': "Ich habe ____ gesehen.",
                        'de_solution': "Ich habe {} gesehen.",
                        'fr_solution': "J'ai vu {}."
                    },
                    {
                        'de': "Sie hat ____ gekauft.",
                        'de_solution': "Sie hat {} gekauft.",
                        'fr_solution': "Elle a acheté {}."
                    }
                ],
                'end': [
                    {
                        'de': "Das gefällt mir sehr ____.",
                        'de_solution': "Das gefällt mir sehr {}.",
                        'fr_solution': "Cela me plaît beaucoup {}."
                    },
                    {
                        'de': "Ich verstehe das ____.",
                        'de_solution': "Ich verstehe das {}.",
                        'fr_solution': "Je comprends le {}."
                    }
                ]
            }
        }

        # Détermine la catégorie du mot
        if word.endswith('en'):
            word_type = 'verbs'
        elif word.startswith(('der', 'die', 'das', 'ein', 'eine')):
            word_type = 'nouns'
        elif len(word) <= 4:
            word_type = 'prepositions'
        else:
            word_type = 'general'

        # Sélectionne les templates appropriés
        word_templates = templates.get(word_type, templates['general'])
        
        # Crée les exemples
        selected_examples = []
        
        # Au moins un exemple de chaque position disponible
        for position in word_templates.keys():
            if position in ['middle', 'end']:  # On évite les positions 'start'
                template = random.choice(word_templates[position])
                # Format solutions with the actual word
                de_solution = template['de_solution'].format(word) if '{}' in template['de_solution'] else template['de_solution']
                fr_solution = template['fr_solution'].format(translation) if '{}' in template['fr_solution'] else template['fr_solution']
                
                selected_examples.append({
                    'sentence': template['de'].replace('____', '___'),
                    'solution': {
                        'de': de_solution,
                        'fr': fr_solution
                    },
                    'hint': f"Le mot manquant est {position=='middle' and 'au milieu' or 'à la fin'}"
                })

        return {
            'type': 'fill_blank',
            'examples': [
                {
                    'question': {
                        'de': example['sentence'],
                        'fr': f"Complétez la phrase avec le mot '{translation}' ({example['hint']})"
                    },
                    'solution': example['solution'],
                    'context': example['hint']
                }
                for example in selected_examples
            ],
            'answer': word,
            'level': self.determine_word_level(word),
            'translation': translation
        }
    
    def create_multiple_choice_exercise(self, word: str, translation: str) -> Dict:
        """Crée un exercice à choix multiples"""
        # Générer des distracteurs
        distractors = random.sample(self.common_distractors, 3)
        while word in distractors:
            distractors = random.sample(self.common_distractors, 3)
        
        choices = [word] + distractors
        random.shuffle(choices)
        correct_index = choices.index(word)
        
        return {
            'type': 'multiple_choice',
            'question': {
                'de': f"Welches Wort bedeutet '{translation}'?",
                'fr': f"Quel mot signifie '{translation}' ?"
            },
            'choices': choices,
            'correct_answer': word,
            'correct_index': correct_index,
            'level': self.determine_word_level(word)
        }
    
    def create_translation_exercise(self, word: str, translation: str) -> Dict:
        """Crée un exercice de traduction"""
        direction = random.choice(['de_to_fr', 'fr_to_de'])
        
        if direction == 'de_to_fr':
            return {
                'type': 'translation',
                'question': {
                    'de': f"Übersetzen Sie: {word}",
                    'fr': f"Traduisez: {word}"
                },
                'answer': translation,
                'source_word': word,
                'direction': 'de_to_fr',
                'level': self.determine_word_level(word)
            }
        else:
            return {
                'type': 'translation',
                'question': {
                    'de': f"Übersetzen Sie: {translation}",
                    'fr': f"Traduisez: {translation}"
                },
                'answer': word,
                'source_word': translation,
                'direction': 'fr_to_de',
                'level': self.determine_word_level(word)
            }
    
    def create_sentence_building_exercise(self, word: str, translation: str) -> Dict:
        """Crée un exercice de construction de phrase"""
        word_pool = [word, 'ist', 'ein', 'der', 'das', 'sehr', 'schön', 'gut']
        random.shuffle(word_pool)
        
        target_sentence = f"Das ist ein {word}."
        
        return {
            'type': 'sentence_building',
            'question': {
                'de': f"Bilden Sie einen Satz mit diesen Wörtern: {', '.join(word_pool)}",
                'fr': f"Formez une phrase avec ces mots: {', '.join(word_pool)}"
            },
            'word_pool': word_pool,
            'target_sentence': target_sentence,
            'answer': word,
            'level': self.determine_word_level(word)
        }
    
    def create_definition_match_exercise(self, word: str, definition: Dict[str, str]) -> Dict:
        """Crée un exercice de correspondance définition-mot"""
        # Nettoyage et validation de la définition
        def clean_definition(text: str) -> str:
            # Supprimer les répétitions de phrases
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            unique_sentences = list(dict.fromkeys(sentences))  # Supprimer les doublons
            
            # Si la définition est trop courte ou vide, utiliser une définition de secours
            if not unique_sentences:
                if word.endswith('en'):
                    return f"Ein Verb, das une wichtige Handlung beschreibt"
                elif word in ['heute', 'morgen', 'jetzt', 'später']:
                    return f"Ein Zeitbegriff, der einen bestimmten Moment beschreibt"
                else:
                    return f"Ein wichtiges Wort in der deutschen Sprache"
            
            # Prendre jusqu'à 2 phrases uniques pour la définition
            clean_text = '. '.join(unique_sentences[:2])
            return clean_text.strip()

        # Nettoyer les définitions
        clean_de = clean_definition(definition['de'])
        clean_fr = clean_definition(definition['fr'])

        # Formater les questions de manière plus naturelle
        question_templates = {
            'de': [
                "Welches Wort wird hier definiert: \"{}\"?",
                "Zu welchem Wort gehört diese Definition: \"{}\"?",
                "Finden Sie das Wort zu dieser Beschreibung: \"{}\""
            ],
            'fr': [
                "Quel mot correspond à cette définition : \"{}\" ?",
                "Trouvez le mot défini par : \"{}\" ?",
                "Identifiez le mot décrit par : \"{}\" ?"
            ]
        }

        # Choisir aléatoirement un template pour chaque langue
        de_template = random.choice(question_templates['de'])
        fr_template = random.choice(question_templates['fr'])

        return {
            'type': 'definition_match',
            'question': {
                'de': de_template.format(clean_de),
                'fr': fr_template.format(clean_fr)
            },
            'definition': {
                'de': clean_de,
                'fr': clean_fr
            },
            'answer': word,
            'level': self.determine_word_level(word)
        }
    
    def create_gender_article_exercise(self, word: str) -> Dict:
        """Crée un exercice sur les articles et le genre"""
        articles = ['der', 'die', 'das']
        # Pour simplifier, on attribue un article au hasard
        correct_article = random.choice(articles)
        
        return {
            'type': 'gender_article',
            'question': {
                'de': f"Welcher Artikel gehört zu '{word}'?",
                'fr': f"Quel article va avec '{word}' ?"
            },
            'choices': articles,
            'answer': correct_article,
            'word': word,
            'level': self.determine_word_level(word)
        }
    
    def create_exercises(self, word: str, translation: str, definition: Dict[str, str]) -> List[Dict]:
        """Crée plusieurs types d'exercices pour un mot"""
        exercises = []
        exercise_creators = [
            self.create_fill_blank_exercise,
            self.create_multiple_choice_exercise,
            self.create_translation_exercise,
            self.create_sentence_building_exercise,
            lambda w, t: self.create_definition_match_exercise(w, definition),
            lambda w, t: self.create_gender_article_exercise(w)
        ]
        
        # Sélectionner aléatoirement des types d'exercices
        selected_creators = random.sample(exercise_creators, 
                                        min(CONFIG['max_exercises_per_word'], len(exercise_creators)))
        
        for creator in selected_creators:
            try:
                if creator == self.create_definition_match_exercise:
                    exercise = creator(word, translation)
                elif creator == self.create_gender_article_exercise:
                    exercise = creator(word, translation)
                else:
                    exercise = creator(word, translation)
                exercises.append(exercise)
            except Exception as e:
                print(f"Erreur création exercice pour '{word}': {e}")
                continue
        
        return exercises
    
    def determine_word_level(self, word: str) -> str:
        """Détermine le niveau de difficulté du mot"""
        if len(word) <= 4:
            return 'beginner'
        elif len(word) <= 8:
            return 'intermediate'
        else:
            return 'advanced'
    
    def process_word(self, word: str) -> Dict:
        """Traite un mot complet avec traduction, définition, exemple et exercices"""
        translation_info = self.get_basic_translation(word)
        definition = self.generate_definition(word)
        example = self.generate_simple_example(word)
        exercises = self.create_exercises(word, translation_info['fr'], definition)
        level = self.determine_word_level(word)
        
        return {
            'word': word,
            'level': level,
            'translation': translation_info,
            'definition': definition,
            'example': example,
            'exercises': exercises,
            'exercise_count': len(exercises),
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
            
            # Calculer les statistiques d'exercices
            total_exercises = sum(entry['exercise_count'] for entry in vocabulary_entries)
            exercise_types_count = {}
            for entry in vocabulary_entries:
                for exercise in entry['exercises']:
                    ex_type = exercise['type']
                    exercise_types_count[ex_type] = exercise_types_count.get(ex_type, 0) + 1
            
            # Ajouter le vocabulaire au JSON
            data['vocabulary'] = vocabulary_entries
            data['vocabulary_stats'] = {
                'total_words': len(vocabulary_entries),
                'total_exercises': total_exercises,
                'exercise_types': exercise_types_count,
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
            print(f"  → {len(vocabulary_entries)} mots traités")
            print(f"  → {total_exercises} exercices générés")
            print(f"  → Types d'exercices: {', '.join(exercise_types_count.keys())}")
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
        description="Transcrit et analyse un fichier audio en allemand avec génération d'exercices diversifiés",
        epilog="Exemple: python script.py audio.mp3 -o ./output --max-exercises 5"
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
        "--max-exercises",
        type=int,
        default=CONFIG['max_exercises_per_word'],
        help=f"Nombre maximum d'exercices par mot (défaut: {CONFIG['max_exercises_per_word']})"
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
    CONFIG['max_exercises_per_word'] = args.max_exercises
    CONFIG['whisper_model'] = args.model
    
    # Traitement
    processor = TranscriptionProcessor()
    success = processor.process_audio_file(args.audio_file, args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()