#!/usr/bin/env python3
import os
import sys
import json
import argparse
from transformers import MarianMTModel, MarianTokenizer

MODEL_PATH = "./opus-mt-de-fr"

def load_translation_model():
    """Load the German to French translation model from local path"""
    print("Loading local translation model...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
        model = MarianMTModel.from_pretrained(MODEL_PATH)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    """Translate text from German to French"""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

def process_json_file(input_file, tokenizer, model):
    """Process a JSON file containing transcription data"""
    print(f"Processing: {input_file}")
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each segment
    for segment in data['segments']:
        # Get the German text
        text = segment['text'].strip()
        
        # Translate to French
        translation = translate_text(text, tokenizer, model)
        
        # Add translation to the segment
        segment['translation'] = translation
    
    # Save the modified JSON
    output_file = input_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Translations saved to: {output_file}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Translate transcribed JSON files from German to French",
        usage="%(prog)s <json_file_or_directory> [options]"
    )
    parser.add_argument(
        "input_path",
        metavar="json_file_or_directory",
        help="Path to JSON file or directory containing JSON files"
    )
    
    args = parser.parse_args()
    input_path = args.input_path
    
    # Load translation model
    tokenizer, model = load_translation_model()
    
    # Process files
    if os.path.isfile(input_path):
        # Single file mode
        if not input_path.lower().endswith('.json'):
            print(f"Error: File '{input_path}' is not a JSON file.", file=sys.stderr)
            sys.exit(1)
        json_files = [input_path]
    else:
        # Directory mode
        if not os.path.isdir(input_path):
            print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
            sys.exit(1)
        print(f"Searching for JSON files in: {input_path}")
        json_files = [
            os.path.join(input_path, f) 
            for f in os.listdir(input_path) 
            if f.lower().endswith('.json')
        ]
    
    if not json_files:
        print("No JSON files found.")
        sys.exit(0)
    
    print(f"\nStarting translation of {len(json_files)} JSON files...")
    print(f"Translation Model Path: {MODEL_PATH}")
    print("-" * 40)
    
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            if process_json_file(json_file, tokenizer, model):
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}", file=sys.stderr)
            error_count += 1
        print("-" * 40)
    
    print("\nTranslation process completed.")
    print(f"Files processed successfully: {success_count}")
    if error_count > 0:
        print(f"Files with errors: {error_count}")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
