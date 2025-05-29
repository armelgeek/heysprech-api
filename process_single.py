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
    'etymology': "Die Etymologie des Wortes '{word}' ist:",
    'definition': "Definition und Verwendung des Wortes '{word}':",
    'expressions': "Häufige Redewendungen mit '{word}':",
    'collocations': "Typische Wortverbindungen mit '{word}':",
    'synonyms': "Synonyme für '{word}':",
    'antonyms': "Antonyme für '{word}':",
    'register': "Sprachregister und Verwendungskontext von '{word}':"
}

# Prompts pour les exercices d'apprentissage
EXERCISE_PROMPTS = {
    'fill_blank': "Generiere einen Lückentextsatz mit dem Wort '{word}':",
    'multiple_choice': "Erstelle Multiple-Choice-Fragen zum Wort '{word}':",
    'word_association': "Erstelle ein Wortassoziationsspiel mit '{word}':",
    'scramble': "Erstelle einen Satz mit '{word}' zum Unscrambling:",
    'context_quiz': "Erstelle Kontextfragen für das Wort '{word}':",
    'usage_quiz': "Erstelle Verwendungsbeispiele mit '{word}' zum Üben:"
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
    """Obtient la traduction en français"""
    translations = {}
    
    # Français uniquement
    tokenizer_de_fr, model_de_fr = models['de_fr']
    inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
    outputs = model_de_fr.generate(**inputs)
    translations['fr'] = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
    
    return translations

def create_lexical_entry(word, models):
    """Crée une entrée lexicographique complète pour un mot"""
    entry = {
        'word': word,
        'grammatical_info': analyze_word_class(word, models),
        'translations': get_translations(word, models),
        'etymology': get_lexical_info(word, 'etymology', models),
        'definitions': get_lexical_info(word, 'definition', models),
        'expressions': get_lexical_info(word, 'expressions', models),
        'collocations': get_lexical_info(word, 'collocations', models),
        'synonyms': get_lexical_info(word, 'synonyms', models),
        'antonyms': get_lexical_info(word, 'antonyms', models),
        'register': get_lexical_info(word, 'register', models),
        'examples': generate_example_sentences(word, models),
        'exercises': generate_exercises(word, models)
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

def generate_exercises(word, models):
    """Génère différents exercices pour l'apprentissage du mot"""
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

if __name__ == "__main__":
    main()
