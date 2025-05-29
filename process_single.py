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
        'beginner': "Écrivez une phrase avec ___ pour '{word}':",
        'intermediate': "Écrivez une phrase avec ___ pour '{word}':",
        'advanced': "Écrivez une phrase complexe avec ___ pour '{word}':"
    },
    'multiple_choice': {
        'beginner': "Quelle est la signification de '{word}'? A) B) C) D):",
        'intermediate': "Dans quel contexte utilise-t-on '{word}'? A) B) C) D):",
        'advanced': "Comment utilise-t-on '{word}' correctement? A) B) C) D):"
    },
    'word_association': {
        'beginner': "Donnez 3 mots associés à '{word}':",
        'intermediate': "Donnez 4 mots du même thème que '{word}':",
        'advanced': "Donnez 5 mots liés au concept de '{word}':"
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
    """Génère une phrase d'exemple simple pour un mot"""
    tokenizer_gpt, model_gpt = models['gpt']
    tokenizer_de_fr, model_de_fr = models['de_fr']
    
    # Un seul prompt simple
    prompt = f"Ich {word}"
    
    try:
        # Génération de la phrase en allemand
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=20,  # Longueur réduite pour une phrase simple
            num_beams=3,
            temperature=0.7,
            do_sample=True
        )
        german = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Traduction directe DE -> FR
        inputs = tokenizer_de_fr(german, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(**inputs)
        french = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        
        return [{
            'de': german,
            'fr': french
        }]  # Retourne une liste avec un seul exemple
    except Exception as e:
        print(f"Erreur lors de la génération d'exemple: {e}")
        return []

def process_vocabulary(text, models):
    """Traite un mot du vocabulaire avec analyse lexicographique complète"""
    print(f"\nAnalyse lexicographique de '{text}'...")
    return create_lexical_entry(text, models)

def analyze_word_class(word, models):
    """Détermine la classe grammaticale et les informations morphologiques"""
    tokenizer_gpt, model_gpt = models['gpt']
    prompt = f"Kurze grammatikalische Analyse für '{word}': Wortart, Genus, Numerus, Kasus."
    
    try:
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=50,  # Réduits pour une réponse concise
            num_beams=3,
            temperature=0.3,  # Température basse pour plus de précision
            do_sample=False  # Pas d'échantillonnage pour plus de précision
        )
        analysis = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Traduction de l'analyse
        tokenizer_de_fr, model_de_fr = models['de_fr']
        inputs = tokenizer_de_fr(analysis, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=3,
            temperature=0.3
        )
        analysis_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'de': analysis,
            'fr': analysis_fr
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
    """Obtient la traduction principale et quelques variantes
    
    Args:
        word: Le mot à traduire
        models: Les modèles de traduction et génération
        
    Returns:
        dict: Traduction principale et quelques variantes
    """
    tokenizer_de_fr, model_de_fr = models['de_fr']
    translations = {
        'principal': '',
        'variantes': []
    }
    
    try:
        # Traduction principale
        inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=20,  # Réduits pour avoir des traductions concises
            num_beams=3,
            temperature=0.3,  # Température basse pour une traduction fidèle
            do_sample=False
        )
        translations['principal'] = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        
        # Quelques variantes (maximum 3)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=20,
            num_return_sequences=3,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        
        for output in outputs:
            translation = tokenizer_de_fr.decode(output, skip_special_tokens=True)
            if translation != translations['principal'] and translation not in translations['variantes']:
                translations['variantes'].append(translation)
                
    except Exception as e:
        print(f"Erreur lors de la traduction: {e}")
        translations['principal'] = word
    
    return translations

def create_lexical_entry(word, models):
    """Crée une entrée lexicographique simplifiée pour un mot
    
    Args:
        word: Le mot à analyser
        models: Les modèles de traduction et génération
        
    Returns:
        dict: Entrée lexicographique avec les informations essentielles
    """
    # Obtenir uniquement les informations essentielles
    translations = get_translations(word, models)
    examples = generate_example_sentences(word, models)
    
    # Structure simplifiée de l'entrée
    entry = {
        'word': word,
        'level': get_word_level(word),
        'translations': translations,
        'examples': examples[:2],  # Limité à 2 exemples
        'exercises': generate_exercises(word, models)
    }
    
    return entry

def format_exercise(exercise_type, exercise_text, word):
    """Formate l'exercice selon son type de manière simplifiée"""
    if exercise_type == 'multiple_choice':
        lines = exercise_text.split('\n')
        return {
            'question': lines[0],
            'choices': [l.strip() for l in lines[1:5] if l.strip()],
            'answer': lines[-1].replace('Correct: ', '') if len(lines) > 5 else 'A'
        }
    elif exercise_type == 'fill_blank':
        return {
            'text': exercise_text.replace(word, '___'),
            'answer': word
        }
    elif exercise_type == 'word_association':
        words = [w.strip() for w in exercise_text.split(',')]
        return {'words': words}
    
    return {'text': exercise_text}

def clean_word(word):
    """Nettoie un mot en retirant la ponctuation et les caractères spéciaux non-allemands"""
    # Garde les caractères allemands spéciaux (umlauts et ß)
    # mais retire la ponctuation et autres caractères spéciaux
    import re
    word = word.strip()
    # Retire la ponctuation à la fin du mot
    word = re.sub(r'[.,!?]$', '', word)
    return word

def generate_exercises(word, models, difficulty='intermediate'):
    """Génère des exercices simplifiés pour l'apprentissage du mot"""
    tokenizer_gpt, model_gpt = models['gpt']
    tokenizer_de_fr, model_de_fr = models['de_fr']
    exercises = {}
    
    clean_word_input = clean_word(word)
    translation = get_translations(clean_word_input, models)
    word_fr = translation.get('principal', clean_word_input)
    
    # Exercice à trous (fill_blank)
    try:
        # Générer 3 phrases simples en allemand
        prompt = f"3 kurze Beispielsätze mit '{clean_word_input}':"
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=100,
            num_beams=3,
            temperature=0.7,
            num_return_sequences=3
        )
        german_sentences = [tokenizer_gpt.decode(output, skip_special_tokens=True).strip() for output in outputs]
        
        # Traduire chaque phrase
        french_sentences = []
        for sent in german_sentences:
            inputs = tokenizer_de_fr(sent, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            french_sentences.append(tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True))
        
        exercises['fill_blank'] = {
            'de': {
                'text': german_sentences[0].replace(clean_word_input, '___'),
                'answer': clean_word_input,
                'examples': german_sentences
            },
            'fr': {
                'text': french_sentences[0].replace(word_fr, '___'),
                'answer': word_fr,
                'examples': french_sentences
            }
        }
        
        # Choix multiples
        prompt = f"4 Bedeutungen für '{clean_word_input}', erste ist korrekt:"
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=100,
            num_beams=4,
            temperature=0.7
        )
        choices_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).split('\n')
        
        # Traduire les choix
        choices_fr = []
        for choice in choices_de:
            if choice.strip():
                inputs = tokenizer_de_fr(choice, return_tensors="pt", padding=True)
                outputs = model_de_fr.generate(**inputs)
                choices_fr.append(tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True))
        
        exercises['multiple_choice'] = {
            'de': {
                'question': f"Was bedeutet '{clean_word_input}'?",
                'choices': choices_de[:4],  # Limiter à 4 choix
                'answer': 'A'  # Le premier choix est toujours correct
            },
            'fr': {
                'question': f"Que signifie '{clean_word_input}'?",
                'choices': choices_fr[:4],  # Limiter à 4 choix
                'answer': 'A'
            }
        }
        
        # Association de mots
        prompt = f"3 verwandte Wörter zu '{clean_word_input}':"
        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=3,
            temperature=0.7
        )
        related_words_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).split(',')
        related_words_de = [w.strip() for w in related_words_de if w.strip()][:3]
        
        # Traduire les mots associés
        related_words_fr = []
        for word in related_words_de:
            inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            related_words_fr.append(tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True))
        
        exercises['word_association'] = {
            'de': {'words': related_words_de},
            'fr': {'words': related_words_fr}
        }
        
    except Exception as e:
        print(f"Erreur lors de la génération des exercices: {e}")
        return None
    
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
