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
DE_EN_MODEL = "Helsinki-NLP/opus-mt-de-en"
EN_FR_MODEL = "Helsinki-NLP/opus-mt-en-fr"

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
    tokenizer_de_fr = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_PATH)
    model_de_fr = MarianMTModel.from_pretrained(TRANSLATION_MODEL_PATH)
    
    # Modèle DE -> EN
    tokenizer_de_en = MarianTokenizer.from_pretrained(DE_EN_MODEL)
    model_de_en = MarianMTModel.from_pretrained(DE_EN_MODEL)
    
    # Modèle EN -> FR
    tokenizer_en_fr = MarianTokenizer.from_pretrained(EN_FR_MODEL)
    model_en_fr = MarianMTModel.from_pretrained(EN_FR_MODEL)
    
    # Modèle de génération allemand
    tokenizer_gpt = AutoTokenizer.from_pretrained(GERMAN_GPT_MODEL)
    model_gpt = GPT2LMHeadModel.from_pretrained(GERMAN_GPT_MODEL)
    tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
    
    return {
        'de_fr': (tokenizer_de_fr, model_de_fr),
        'de_en': (tokenizer_de_en, model_de_en),
        'en_fr': (tokenizer_en_fr, model_en_fr),
        'gpt': (tokenizer_gpt, model_gpt)
    }

def generate_example_sentences(word, models):
    """Génère des phrases d'exemple pour un mot"""
    tokenizer_gpt, model_gpt = models['gpt']
    
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
            
            # Traduction DE -> EN -> FR pour plus de précision
            tokenizer_de_en, model_de_en = models['de_en']
            tokenizer_en_fr, model_en_fr = models['en_fr']
            
            # DE -> EN
            inputs = tokenizer_de_en(german, return_tensors="pt", padding=True)
            outputs = model_de_en.generate(**inputs)
            english = tokenizer_de_en.decode(outputs[0], skip_special_tokens=True)
            
            # EN -> FR
            inputs = tokenizer_en_fr(english, return_tensors="pt", padding=True)
            outputs = model_en_fr.generate(**inputs)
            french = tokenizer_en_fr.decode(outputs[0], skip_special_tokens=True)
            
            examples.append({
                'de': german,
                'en': english,
                'fr': french
            })
        except Exception as e:
            print(f"Erreur lors de la génération d'exemple pour '{prompt}': {e}")
            continue
            
    return examples

def process_vocabulary(text, models):
    """Traite un mot du vocabulaire"""
    tokenizer_de_fr, model_de_fr = models['de_fr']
    
    # Traduction directe
    inputs = tokenizer_de_fr(text, return_tensors="pt", padding=True)
    outputs = model_de_fr.generate(**inputs)
    translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
    
    # Génération d'exemples
    examples = generate_example_sentences(text, models)
    
    return {
        'word': text,
        'translation': translation,
        'examples': examples
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

if __name__ == "__main__":
    main()
