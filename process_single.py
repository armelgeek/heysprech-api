#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from transformers import (
    MarianMTModel, 
    MarianTokenizer, 
    GPT2LMHeadModel, 
    AutoTokenizer
)
from tqdm import tqdm
import json

# Configuration
MODEL = "base"
LANGUAGE = "de"
OUTPUT_FORMAT = "json"
TRANSLATION_MODEL_PATH = "./opus-mt-de-fr"
GERMAN_GPT_MODEL = "benjamin/gpt2-wechsel-german"

# Prompts pour l'analyse lexicographique
LEXICAL_PROMPTS = {
    'etymology': "Die Etymologie des Wortes '{word}' (mit allen historischen Entwicklungen, Wurzeln und Spracheinflüssen):",
    'definition': "Gib eine sehr detaillierte Definition des Wortes '{word}' mit den folgenden Aspekten:\n1. Hauptbedeutung mit präziser Erklärung\n2. Alle Nebenbedeutungen und Nuancen\n3. Spezielle Verwendungen und Kontexte\n4. Fachsprachliche Bedeutungen und Terminologie\n5. Konkrete Beispiele für jeden Aspekt\n6. Grammatikalische Besonderheiten\n7. Kulturelle oder regionale Besonderheiten",
    'detailed_usage': "Beschreibe sehr detailliert und mit vielen Beispielen, wie das Wort '{word}' verwendet wird in:\n1. Alltäglicher Sprache und Umgangssprache\n2. Formeller Sprache und gehobener Sprache\n3. Fachsprache und spezifischen Bereichen\n4. Regionalen Varianten und Dialekten\n5. Literarischer Sprache\n6. Modernde Medien und Jugendsprache",
    'expressions': "Liste ALLE Redewendungen, Sprichwörter und feste Ausdrücke mit '{word}' auf. Für jeden Ausdruck gib an:\n1. Wörtliche Bedeutung\n2. Übertragene Bedeutung\n3. Verwendungskontext\n4. Stilebene\n5. Regionale Besonderheiten\n6. Beispiele in Sätzen",
    'collocations': "Liste ALLE üblichen Wortverbindungen mit '{word}' auf, gruppiert nach:\n1. Verb + {word}\n2. {word} + Substantiv\n3. Adjektiv + {word}\n4. Präposition + {word}\n5. Typische Satzstrukturen\nFür jede Kombination gib mehrere Beispiele und Kontextinformationen.",
    'synonyms': "Liste ALLE möglichen Synonyme für '{word}' auf, sehr detailliert geordnet nach:\n1. Exakte Synonyme (gleiche Bedeutung und Verwendung)\n2. Kontextabhängige Synonyme (mit Erklärung der Nuancen)\n3. Stilistische Varianten (formell, umgangssprachlich, etc.)\n4. Regionale Varianten und Dialektausdrücke\n5. Fachsprachliche Alternativen\nFüge für JEDES Synonym eine Erklärung der Bedeutungsnuancen und Verwendungsbeispiele hinzu.",
    'antonyms': "Liste alle möglichen Gegenwörter zu '{word}' auf, geordnet nach:\n1. Direkte Antonyme\n2. Kontextabhängige Gegensätze\n3. Graduelle Abstufungen\nFür jedes Antonym erkläre die genaue Art des Gegensatzes und gib Verwendungsbeispiele.",
    'register': "Beschreibe sehr detailliert das Sprachregister und die Verwendungskontexte von '{word}', einschließlich:\n1. Stilebene (formell, neutral, umgangssprachlich, etc.)\n2. Soziale Kontexte\n3. Situative Angemessenheit\n4. Historische Entwicklung des Gebrauchs\n5. Aktuelle Trends in der Verwendung\n6. Regionale und soziale Variation",
    'semantic_field': "Beschreibe das semantische Feld und verwandte Begriffe zu '{word}' mit:\n1. Übergeordnete Begriffe (Hyperonyme)\n2. Untergeordnete Begriffe (Hyponyme)\n3. Verwandte Konzepte\n4. Typische Assoziationen\n5. Kulturelle Konnotationen\n6. Semantische Beziehungen zu anderen Wortfeldern"
}

# Prompts pour les exercices d'apprentissage
EXERCISE_PROMPTS = {
    'fill_blank': {
        'beginner': "Generiere einen einfachen Lückentextsatz mit dem Wort '{word}':",
        'intermediate': "Generiere einen mittelschweren Lückentextsatz mit dem Wort '{word}':",
        'advanced': "Generiere einen komplexen Lückentextsatz mit dem Wort '{word}':"
    },
    'multiple_choice': {
        'beginner': "Erstelle einfache Multiple-Choice-Fragen zum Wort '{word}' mit Grundbedeutungen:",
        'intermediate': "Erstelle Multiple-Choice-Fragen zum Wort '{word}' mit verschiedenen Kontexten:",
        'advanced': "Erstelle anspruchsvolle Multiple-Choice-Fragen zum Wort '{word}' mit Nuancen und idiomatischen Verwendungen:"
    },
    'word_association': {
        'beginner': "Erstelle ein einfaches Wortassoziationsspiel mit '{word}' (max. 4 Wörter):",
        'intermediate': "Erstelle ein Wortassoziationsspiel mit '{word}' und thematischen Gruppen:",
        'advanced': "Erstelle ein komplexes Wortassoziationsnetz mit '{word}' und semantischen Beziehungen:"
    },
    'scramble': {
        'beginner': "Erstelle einen einfachen Satz mit '{word}' (max. 5 Wörter) zum Unscrambling:",
        'intermediate': "Erstelle einen Satz mittlerer Länge mit '{word}' zum Unscrambling:",
        'advanced': "Erstelle einen komplexen Satz mit '{word}' und Nebensätzen zum Unscrambling:"
    },
    'context_quiz': {
        'beginner': "Erstelle einfache Kontextfragen für Anfänger zum Wort '{word}':",
        'intermediate': "Erstelle Kontextfragen mittleren Niveaus zum Wort '{word}':",
        'advanced': "Erstelle anspruchsvolle Kontextfragen mit kulturellen Bezügen zum Wort '{word}':"
    },
    'usage_quiz': {
        'beginner': "Erstelle grundlegende Verwendungsbeispiele mit '{word}':",
        'intermediate': "Erstelle Verwendungsbeispiele mit '{word}' in verschiedenen Zeitformen:",
        'advanced': "Erstelle Verwendungsbeispiele mit '{word}' in idiomatischen Wendungen:"
    }
}

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

def transcribe_file(audio_path, output_directory):
    """Transcribe a single audio file"""
    print(f"Transcribing: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}.{OUTPUT_FORMAT}"
    output_file_path = os.path.join(output_directory, output_filename)

    command = [
        sys.executable, "-m", "whisperx",
        audio_path,
        "--model", MODEL,
        "--language", LANGUAGE,
        "--output_format", OUTPUT_FORMAT,
        "--output_dir", output_directory,
        "--segment_resolution", "chunk",
        "--max_line_count", "1",
        "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
        "--compute_type", "float32",
        "--max_line_width", "-50"
    ]

    try:
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            print(f"✓ Transcription completed")
            return output_file_path
        else:
            print(f"✗ Transcription failed with code {result.returncode}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"✗ Error during transcription: {e}", file=sys.stderr)
        return None

def translate_json(json_path, models):
    """Translate transcribed JSON file and process vocabulary"""
    print(f"Translating and processing vocabulary: {json_path}")
    
    try:
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer_de_fr, model_de_fr = models['de_fr']
        
        # Collecter le vocabulaire unique
        all_words = set()
        for segment in data['segments']:
            text = segment['text'].strip()
            # Traduire le segment
            inputs = tokenizer_de_fr(text, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            segment['translation'] = translation
            
            # Ajouter les mots au vocabulaire
            words = text.split()
            all_words.update(word.lower() for word in words)
        
        # Traiter chaque mot du vocabulaire
        print("\nTraitement du vocabulaire...")
        vocabulary = []
        for word in tqdm(sorted(all_words), desc="Mots"):
            vocab_entry = process_vocabulary(word, models)
            vocabulary.append(vocab_entry)
        
        # Ajouter la section vocabulaire au JSON
        data['vocabulary'] = vocabulary
        
        # Save modified JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("✓ Translation et vocabulaire complétés")
        return True
    except Exception as e:
        print(f"✗ Error during translation: {e}", file=sys.stderr)
        return False

def load_models():
    """Charge tous les modèles nécessaires"""
    print("Chargement des modèles de traduction et génération...")
    
    # Modèle DE -> FR direct
    print("✓ Chargement modèle DE -> FR...")
    tokenizer_de_fr = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
    model_de_fr = MarianMTModel.from_pretrained(TRANSLATION_MODEL_PATH)
    
    # Modèle de génération allemand
    print("✓ Chargement modèle GPT allemand...")
    tokenizer_gpt = AutoTokenizer.from_pretrained(GERMAN_GPT_MODEL)
    model_gpt = GPT2LMHeadModel.from_pretrained(GERMAN_GPT_MODEL)
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    
    return {
        'de_fr': (tokenizer_de_fr, model_de_fr),
        'gpt': (tokenizer_gpt, model_gpt)
    }

def generate_example_sentences(word, models):
    """Génère des phrases d'exemple pour un mot"""
    tokenizer_gpt, model_gpt = models['gpt']
    tokenizer_de_fr, model_de_fr = models['de_fr']
    
    prompts = [
        f"Das {word} ist",
        f"Ich {word}",
        f"Der {word} hat",
        f"Mit {word} kann man"
    ]
    
    examples = []
    for prompt in prompts:
        try:
            # Génération de la phrase en allemand
            inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True
            )
            german = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
            
            # Traduction directe DE -> FR
            inputs = tokenizer_de_fr(german, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            french = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            
            examples.append({
                'de': german,
                'fr': french
            })
        except Exception as e:
            print(f"Erreur lors de la génération d'exemple pour '{prompt}': {e}")
            continue
            
    return examples

def process_vocabulary(text, models):
    """Traite un mot du vocabulaire avec analyse lexicographique complète"""
    print(f"\nAnalyse lexicographique de '{text}'...")
    return create_lexical_entry(text, models)

def analyze_word_class(word, models):
    """Détermine la classe grammaticale et les informations morphologiques"""
    tokenizer_gpt, model_gpt = models['gpt']
    prompt = f"Grammatische Analyse des Wortes '{word}':"
    
    try:
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=100,
            num_beams=5,
            temperature=0.7
        )
        analysis = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Traduction de l'analyse
        tokenizer_de_fr, model_de_fr = models['de_fr']
        inputs = tokenizer_de_fr(analysis, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(**inputs)
        analysis_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'original': analysis,
            'translation': analysis_fr
        }
    except Exception as e:
        print(f"Erreur lors de l'analyse grammaticale de '{word}': {e}")
        return None

def get_lexical_info(word, prompt_key, models):
    """Obtient des informations lexicales spécifiques pour un mot"""
    tokenizer_gpt, model_gpt = models['gpt']
    prompt = LEXICAL_PROMPTS[prompt_key].format(word=word)
    
    try:
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=5,
            temperature=0.7
        )
        info = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Traduction
        tokenizer_de_fr, model_de_fr = models['de_fr']
        inputs = tokenizer_de_fr(info, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(**inputs)
        info_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'de': info,
            'fr': info_fr
        }
    except Exception as e:
        print(f"Erreur lors de la récupération des informations '{prompt_key}' pour '{word}': {e}")
        return None

def get_translations(word, models):
    """Obtient plusieurs variantes de traduction en français avec différents niveaux de style
    
    Args:
        word: Le mot à traduire
        models: Les modèles de traduction et génération
        
    Returns:
        dict: Traductions structurées par style et contexte
    """
    tokenizer_de_fr, model_de_fr = models['de_fr']
    translations = {
        'principal': '',  # Traduction principale/neutre
        'styles': {       # Variations stylistiques
            'formel': [],
            'courant': [],
            'familier': []
        },
        'contextes': {},  # Traductions selon le contexte
        'alternatives': [] # Autres variantes possibles
    }
    
    # Configurations pour différents styles de traduction
    generation_params = [
        # (température, top_k, top_p) - plus la température est haute, plus les résultats sont créatifs
        (0.3, 50, 0.95),  # Style neutre/standard
        (0.7, 50, 0.95),  # Style varié
        (0.9, 50, 0.95)   # Style très créatif
    ]
    
    # Génération de la traduction principale (style neutre)
    try:
        inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=5,
            temperature=0.3,  # Température basse pour une traduction fidèle
            do_sample=False   # Pas d'échantillonnage pour la traduction principale
        )
        translations['principal'] = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Erreur lors de la traduction principale: {e}")
        translations['principal'] = word
    
    # Génération des variations stylistiques
    contexts = [
        ("in formellen Texten", 'formel'),
        ("in der Alltagssprache", 'courant'),
        ("in der Umgangssprache", 'familier')
    ]
    
    for context, style in contexts:
        prompt = f"Das Wort '{word}' {context}:"
        try:
            inputs = tokenizer_de_fr(prompt, return_tensors="pt", padding=True)
            
            # Utiliser différents paramètres selon le style
            if style == 'formel':
                temp, k, p = 0.3, 50, 0.95  # Plus conservateur
            elif style == 'familier':
                temp, k, p = 0.9, 50, 0.95  # Plus créatif
            else:
                temp, k, p = 0.7, 50, 0.95  # Intermédiaire
                
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=3,
                temperature=temp,
                top_k=k,
                top_p=p,
                do_sample=True
            )
            
            for output in outputs:
                translation = tokenizer_de_fr.decode(output, skip_special_tokens=True)
                if translation not in translations['styles'][style]:
                    translations['styles'][style].append(translation)
                    
        except Exception as e:
            print(f"Erreur lors de la génération du style {style}: {e}")
    
    # Génération de traductions contextuelles
    specific_contexts = [
        "in wissenschaftlichen Texten",
        "in der Literatur",
        "in der Technik",
        "in der Wirtschaft",
        "in den Medien"
    ]
    
    for context in specific_contexts:
        prompt = f"Das Wort '{word}' {context}:"
        try:
            inputs = tokenizer_de_fr(prompt, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            
            context_key = context.split()[-1].rstrip(':')  # Extrait le contexte principal
            translations['contextes'][context_key] = [
                tokenizer_de_fr.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
        except Exception as e:
            print(f"Erreur lors de la génération du contexte {context}: {e}")
    
    # Génération d'alternatives générales
    for temp, k, p in generation_params:
        try:
            inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=50,
                num_return_sequences=3,
                temperature=temp,
                top_k=k,
                top_p=p,
                do_sample=True
            )
            
            for output in outputs:
                translation = tokenizer_de_fr.decode(output, skip_special_tokens=True)
                if (translation != translations['principal'] and 
                    translation not in translations['alternatives']):
                    translations['alternatives'].append(translation)
                    
        except Exception as e:
            print(f"Erreur lors de la génération d'alternatives: {e}")
    
    return translations

def create_lexical_entry(word, models):
    """Crée une entrée lexicographique complète pour un mot
    
    Args:
        word: Le mot à analyser
        models: Les modèles de traduction et génération
        
    Returns:
        dict: Entrée lexicographique complète avec toutes les informations
    """
    # Obtenir la fréquence et le niveau de difficulté recommandé
    word_level = get_word_level(word)
    
    # Générer toutes les informations lexicales
    translations = get_translations(word, models)
    definitions = get_lexical_info(word, 'definition', models)
    examples = generate_example_sentences(word, models)
    pronunciation = get_pronunciation(word, models)
    
    # Structure enrichie de l'entrée lexicographique
    entry = {
        'word': word,
        'level': word_level,
        'pronunciation': pronunciation,  # Ajout de la prononciation
        'translations': translations,
        'grammatical_info': analyze_word_class(word, models),
        'definitions': {
            'detaillees': definitions,
            'exemples': examples
        },
        'etymologie': get_lexical_info(word, 'etymology', models),
        'usage': {
            'general': get_lexical_info(word, 'detailed_usage', models),
            'expressions': get_lexical_info(word, 'expressions', models),
            'collocations': get_lexical_info(word, 'collocations', models),
            'registre': get_lexical_info(word, 'register', models)
        },
        'semantic': {
            'synonymes': get_lexical_info(word, 'synonyms', models),
            'antonymes': get_lexical_info(word, 'antonyms', models),
            'champ_semantique': get_lexical_info(word, 'semantic_field', models)
        },
        'exercices': generate_exercises(word, models, difficulty=word_level['level'])
    }
    
    return entry

def format_exercise(exercise_type, exercise_text, word):
    """Formate l'exercice selon son type"""
    if exercise_type == 'multiple_choice':
        # Format attendu: Question\nA) réponse1\nB) réponse2\nC) réponse3\nD) réponse4\nCorrect: X
        lines = exercise_text.split('\n')
        if len(lines) >= 5:  # Au moins une question et 4 choix
            return {
                'question': lines[0],
                'choices': [
                    {'id': 'A', 'text': lines[1][3:] if lines[1].startswith('A)') else lines[1]},
                    {'id': 'B', 'text': lines[2][3:] if lines[2].startswith('B)') else lines[2]},
                    {'id': 'C', 'text': lines[3][3:] if lines[3].startswith('C)') else lines[3]},
                    {'id': 'D', 'text': lines[4][3:] if lines[4].startswith('D)') else lines[4]}
                ],
                'correct_answer': 'B'  # Par défaut, la bonne réponse est la traduction correcte
            }
    elif exercise_type == 'fill_blank':
        # Format: Phrase avec ___ pour le mot manquant
        return {
            'sentence': exercise_text.replace(word, '___'),
            'answer': word
        }
    elif exercise_type == 'scramble':
        # Format: Mots mélangés | Phrase correcte
        if '|' in exercise_text:
            scrambled, correct = exercise_text.split('|')
            return {
                'scrambled_words': scrambled.strip().split(),
                'correct_sentence': correct.strip()
            }
    elif exercise_type == 'word_association':
        # Format: mot1:catégorie1, mot2:catégorie2, etc.
        pairs = [pair.strip() for pair in exercise_text.split(',')]
        return {
            'pairs': [{'word': p.split(':')[0].strip(), 'category': p.split(':')[1].strip()} 
                     for p in pairs if ':' in p]
        }
    elif exercise_type == 'context_quiz':
        # Format: Question?|Réponse correcte
        if '|' in exercise_text:
            question, answer = exercise_text.split('|')
            return {
                'question': question.strip(),
                'correct_answer': answer.strip()
            }
    
    # Format par défaut
    return {'text': exercise_text}

def generate_exercises(word, models, difficulty='intermediate'):
    """Génère différents exercices pour l'apprentissage du mot
    
    Args:
        word: Le mot à traiter
        models: Les modèles de traduction et génération
        difficulty: Niveau de difficulté ('beginner', 'intermediate', 'advanced')
    """
    tokenizer_gpt, model_gpt = models['gpt']
    tokenizer_de_fr, model_de_fr = models['de_fr']
    exercises = {}
    
    # Génération des différents types d'exercices
    for exercise_type, prompt in EXERCISE_PROMPTS.items():
        try:
            # Génération de l'exercice en allemand
            full_prompt = prompt.format(word=word)
            inputs = tokenizer_gpt(full_prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=5,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            exercise_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
            
            # Traduction en français
            inputs = tokenizer_de_fr(exercise_de, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            exercise_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            
            # Formatage structuré des exercices
            formatted_de = format_exercise(exercise_type, exercise_de, word)
            formatted_fr = format_exercise(exercise_type, exercise_fr, 
                                        get_translations(word, models)['fr'])
            
            exercises[exercise_type] = {
                'de': formatted_de,
                'fr': formatted_fr,
                'type': exercise_type
            }
            
        except Exception as e:
            print(f"Erreur lors de la génération de l'exercice '{exercise_type}' pour '{word}': {e}")
            exercises[exercise_type] = None
    
    return exercises

def get_word_level(word):
    """Détermine le niveau de difficulté d'un mot allemand
    
    Basé sur :
    - La longueur du mot
    - La présence de caractères spéciaux
    - La complexité morphologique
    """
    # Liste de préfixes et suffixes courants en allemand
    common_prefixes = ['un', 'auf', 'ab', 'ein', 'aus', 'mit', 'vor']
    common_suffixes = ['ung', 'heit', 'keit', 'lich', 'ig', 'isch']
    
    # Calcul du score de complexité
    score = 0
    
    # Longueur du mot
    if len(word) <= 4:
        score += 1
    elif len(word) <= 7:
        score += 2
    else:
        score += 3
    
    # Caractères spéciaux (umlauts, ß)
    special_chars = 'äöüßÄÖÜ'
    if any(c in word for c in special_chars):
        score += 1
    
    # Complexité morphologique
    if any(word.startswith(prefix) for prefix in common_prefixes):
        score += 1
    if any(word.endswith(suffix) for suffix in common_suffixes):
        score += 1
    
    # Détermination du niveau
    if score <= 2:
        return {
            'level': 'beginner',
            'score': score,
            'explanation': 'Mot simple, approprié pour les débutants'
        }
    elif score <= 4:
        return {
            'level': 'intermediate',
            'score': score,
            'explanation': 'Mot de difficulté moyenne'
        }
    else:
        return {
            'level': 'advanced',
            'score': score,
            'explanation': 'Mot complexe, niveau avancé'
        }

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe and translate a single audio file",
        usage="%(prog)s <audio_file> [options]"
    )
    parser.add_argument(
        "audio_file",
        help="Audio file to process"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (required)",
        required=True
    )

    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if not args.audio_file.lower().endswith(AUDIO_EXTENSIONS):
        print(f"Error: Unsupported audio format. Supported formats: {', '.join(AUDIO_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    # Setup output directory
    output_dir = args.output or os.path.dirname(args.audio_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    print("\n1. Starting transcription...")
    json_path = transcribe_file(args.audio_file, output_dir)
    if not json_path:
        print("✗ Transcription failed", file=sys.stderr)
        sys.exit(1)

    print("\n2. Loading models...")
    try:
        models = load_models()
    except Exception as e:
        print(f"✗ Error loading models: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n3. Starting translation and vocabulary processing...")
    if not translate_json(json_path, models):
        print("✗ Translation and vocabulary processing failed", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Processing completed successfully!")
    print(f"Output file: {json_path}")

def get_pronunciation(word, models):
    """Obtient la prononciation phonétique d'un mot allemand
    
    Args:
        word: Le mot dont on veut obtenir la prononciation
        models: Les modèles de traduction et génération
        
    Returns:
        dict: Informations de prononciation incluant IPA et description
    """
    tokenizer_gpt, model_gpt = models['gpt']
    
    # Prompt pour obtenir la prononciation détaillée
    prompt = f"""Geben Sie folgende Details zur Aussprache des Wortes '{word}':
1. IPA-Transkription in eckigen Klammern []
2. Silbentrennung mit Bindestrichen (-)
3. Betonung (markiert mit ' vor der betonten Silbe)
4. Ausspracheregeln und Besonderheiten
5. Regionale Varianten (wenn vorhanden)"""
    
    try:
        # Génération de l'analyse de prononciation
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=5,
            temperature=0.3,  # Température basse pour plus de précision
            top_p=0.95,
            do_sample=False  # Pas d'échantillonnage pour la précision
        )
        pronunciation_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Traduction de l'explication
        tokenizer_de_fr, model_de_fr = models['de_fr']
        inputs = tokenizer_de_fr(pronunciation_de, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(**inputs)
        pronunciation_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        
        # Extraction des informations
        pronunciation_info = {
            'text': word,
            'ipa': extract_ipa(pronunciation_de),
            'syllables': extract_syllables(pronunciation_de),
            'stress': extract_stress_position(pronunciation_de),
            'description': {
                'de': pronunciation_de,
                'fr': pronunciation_fr
            },
            'audio': {
                'url': f"https://api.heysprech.de/audio/pronunciation/{word}.mp3",  # URL fictive
                'speaker': "native_de",
                'dialect': "standard"
            }
        }
        
        return pronunciation_info
        
    except Exception as e:
        print(f"Erreur lors de la génération de la prononciation pour '{word}': {e}")
        return None

def extract_ipa(text):
    """Extrait les symboles IPA du texte de prononciation"""
    import re
    ipa_match = re.search(r'\[(.*?)\]', text)
    if ipa_match:
        return ipa_match.group(1).strip()
    return None

def extract_syllables(text):
    """Extrait la séparation en syllabes du texte"""
    import re
    # Recherche les mots avec des tirets ou points
    syllables_match = re.search(r'(?:Silbentrennung|Silben):\s*([\w\-·]+)', text, re.IGNORECASE)
    if syllables_match:
        # Divise aux tirets ou points
        syllables = re.split(r'[-·]', syllables_match.group(1))
        return [s.strip() for s in syllables if s.strip()]
    return None

def extract_stress_position(text):
    """Extrait la position de l'accent tonique"""
    import re
    # Cherche des indications de l'accent tonique
    stress_match = re.search(r'(?:Betonung|betont):\s*(.+?)(?:\.|$)', text, re.IGNORECASE | re.MULTILINE)
    if stress_match:
        position = stress_match.group(1).strip()
        # Convertir en format standardisé
        if 'erst' in position.lower():
            return 'first'
        elif 'zweit' in position.lower():
            return 'second'
        elif 'letzt' in position.lower():
            return 'last'
        return position
    return None

if __name__ == "__main__":
    main()
