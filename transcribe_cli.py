#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

MODEL = "base"
LANGUAGE = "de"
OUTPUT_FORMAT = "json"

AUDIO_EXTENSIONS = (
    ".opus", ".mp3", ".wav", ".m4a", ".ogg",
    ".flac", ".aac", ".aiff", ".wma"
)

def find_audio_files(directory):
    audio_files = []
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(file_path)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error while searching for files in '{directory}': {e}", file=sys.stderr)
        sys.exit(1)
    return audio_files

def transcribe_file(audio_path, output_directory):
    print(f"Processing: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}.{OUTPUT_FORMAT}"
    output_file_path = os.path.join(output_directory, output_filename)

    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
        print(f"File '{output_filename}' already exists and is not empty. Skipping.")
        return True

    command = [
        sys.executable,
        "-m",
        "whisperx",
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
            print(f"Transcription of '{os.path.basename(audio_path)}' completed.")
            return True
        else:
            print(f"ERROR during transcription of '{os.path.basename(audio_path)}'. Return code: {result.returncode}", file=sys.stderr)
            if os.path.exists(output_file_path):
                try:
                    os.remove(output_file_path)
                    print(f"Incomplete output file '{output_filename}' deleted.")
                except OSError as e:
                    print(f"Avertissement: Impossible de supprimer le fichier de sortie incomplet '{nom_fichier_sortie}': {e}", file=sys.stderr)
            return False

    except ModuleNotFoundError:
        print(f"Erreur: Le module 'whisper' ne semble pas être installé dans l'environnement Python actuel: {sys.executable}", file=sys.stderr)
        print(f"Exécutez: pip install -U openai-whisper", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Erreur inattendue lors de l'exécution de Whisper pour '{chemin_audio}': {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description=f"Transcribe audio file or directory using Whisper (Model: {MODEL}).",
        usage="%(prog)s <audio_file_or_directory> [options]"
    )
    parser.add_argument(
        "input_path",
        metavar="audio_file_or_directory",
        help="Path to an audio file or directory containing audio files"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for JSON files (default: same as input)",
        default=None
    )

    args = parser.parse_args()
    input_path = args.input_path
    
    # Setup output directory
    if args.output:
        output_directory = args.output
        os.makedirs(output_directory, exist_ok=True)
    else:
        output_directory = None  # Will be set based on input type
    
    # Determine if input is a file or directory
    if os.path.isfile(input_path):
        # Single file mode
        if not input_path.lower().endswith(AUDIO_EXTENSIONS):
            print(f"Error: File '{input_path}' is not a supported audio file.", file=sys.stderr)
            print(f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}", file=sys.stderr)
            sys.exit(1)
        audio_files = [input_path]
        output_directory = output_directory or os.path.dirname(input_path) or "."
    else:
        # Directory mode
        if not os.path.isdir(input_path):
            print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
            sys.exit(1)
        print(f"Searching for audio files ({', '.join(AUDIO_EXTENSIONS)}) in: {input_path}")
        audio_files = find_audio_files(input_path)
        output_directory = input_path

    if not audio_files:
        print(f"No audio files found in the specified format.")
        sys.exit(0)

    print(f"\nStarting transcription of {len(audio_files)} audio files found...")
    print(f"Whisper Model: {MODEL}")
    print(f"Language: {LANGUAGE}")
    print(f"Output Format: {OUTPUT_FORMAT}")
    print(f"Output Directory: {output_directory}")
    if MODEL.lower() == "large":
        print("WARNING: The 'large' model requires significant computing resources (RAM/VRAM) and will take time.")
    print("-" * 40)

    success_count = 0
    error_count = 0

    for audio_file in audio_files:
        if transcribe_file(audio_file, output_directory):
            success_count += 1
        else:
            error_count += 1
        print("-" * 40)

    print("Transcription process completed.")
    print(f"Files processed successfully: {success_count}")
    if error_count > 0:
        print(f"Files with transcription errors: {error_count}")

    if error_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()