#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import re  # Ajout de l'import manquant
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

def remove_repetitions(text):
    """Supprime les répétitions de phrases et de mots consécutifs"""
    # Nettoie d'abord les répétitions au niveau des phrases
    sentences = text.split('.')
    clean_sentences = []
    for sent in sentences:
        if sent.strip() and sent.strip() not in clean_sentences:
            clean_sentences.append(sent.strip())
    
    # Nettoie les répétitions de mots consécutifs dans chaque phrase
    final_sentences = []
    for sent in clean_sentences:
        words = sent.split()
        clean_words = [words[i] for i in range(len(words)) 
                      if i == 0 or words[i] != words[i-1]]
        final_sentences.append(' '.join(clean_words))
    
    return '. '.join(final_sentences).strip()

def generate_example_sentences(word, models):
    """Génère deux phrases d'exemple simples pour un mot"""
    tokenizer_gpt, model_gpt = models['gpt']
    tokenizer_de_fr, model_de_fr = models['de_fr']
    examples = []
    
    # Prompts pour quatre phrases courtes et simples
    prompts = [
        f"Ich {word} gern",           # Premier exemple simple
        f"Das {word} ist gut",        # Deuxième exemple descriptif
        f"Wir haben {word}",          # Troisième exemple possessif
        f"Er {word} heute"            # Quatrième exemple temporel
    ]
    
    for prompt in prompts:
        try:
            # Génération en allemand
            inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=15,   # Plus court pour des phrases concises
                num_beams=3,     # Moins de beams pour plus de simplicité
                temperature=0.3, # Température plus basse pour plus de cohérence
                no_repeat_ngram_size=2,  # Éviter les répétitions
                do_sample=False  # Pas d'échantillonnage pour plus de contrôle
            )
            german = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).strip()
            if not german.endswith('.'):
                german += '.'
            
            # Traduction en français
            inputs = tokenizer_de_fr(german, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            french = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True).strip()
            if not french.endswith('.'):
                french += '.'
            
            examples.append({
                'de': german,
                'fr': french
            })
            
        except Exception as e:
            print(f"Erreur lors de la génération d'exemple: {e}")
            continue
    
    # Assurer qu'on a toujours 2 exemples
    while len(examples) < 2:
        examples.append({
            'de': f"Ich {word}.",
            'fr': "Je suis."
        })
    
    return examples

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
        reformulated_fr = reformat_french_text(analysis_fr, models, context='definition')
        
        return {
            'de': analysis,
            'fr': reformulated_fr
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
        reformulated_fr = reformat_french_text(info_fr, models)
        
        return {
            'de': info,
            'fr': reformulated_fr
        }
    except Exception as e:
        print(f"Erreur lors de la récupération des informations '{prompt_key}' pour '{word}': {e}")
        return None

def get_translations(word, models):
    """Obtient la traduction principale, les variantes et les synonymes
    
    Args:
        word: Le mot à traduire
        models: Les modèles de traduction et génération
        
    Returns:
        dict: Traduction principale, variantes et synonymes
    """
    tokenizer_de_fr, model_de_fr = models['de_fr']
    tokenizer_gpt, model_gpt = models['gpt']
    translations = {
        'de': {
            'principal': word,
            'variantes': [],
            'synonymes': [],
            'definition': ''
        },
        'fr': {
            'principal': '',
            'variantes': [],
            'synonymes': [],
            'definition': ''
        }
    }
    
    try:
        # Traduction principale
        inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=20,
            num_beams=3,
            temperature=0.3,
            do_sample=False
        )
        translated = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        translations['fr']['principal'] = reformat_french_text(translated, models, context='definition')
        
        # Variantes de traduction (5+)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=20,
            num_return_sequences=5,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            diversity_penalty=0.5  # Pour encourager la diversité
        )
        
        for output in outputs:
            translation = tokenizer_de_fr.decode(output, skip_special_tokens=True)
            reformulated = reformat_french_text(translation, models)
            if reformulated != translations['fr']['principal'] and reformulated not in translations['fr']['variantes']:
                translations['fr']['variantes'].append(reformulated)
        
        # Générer des synonymes en français
        if translations['principal']:
            prompt = f"Donnez 5 synonymes du mot français '{translations['fr']['principal']}' (un par ligne):"
            inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=5,
                temperature=0.7,
                do_sample=True
            )
            synonyms_text = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
            translations['fr']['synonymes'] = [s.strip() for s in synonyms_text.split('\n') if s.strip()][:5]

            # Générer une définition en français
            prompt_def_fr = f"Définissez le mot '{translations['fr']['principal']}' en une phrase claire et concise:"
            inputs = tokenizer_gpt(prompt_def_fr, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=3,
                temperature=0.3,
                do_sample=False
            )
            translations['fr']['definition'] = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).strip()

            # Générer une définition en allemand
            prompt_def_de = f"Definieren Sie das Wort '{word}' in einem klaren und präzisen Satz:"
            inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=3,
                temperature=0.3,
                do_sample=False
            )
            translations['de']['definition'] = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        print(f"Erreur lors de la traduction: {e}")
        translations['fr']['principal'] = word
    
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
    
    # Structure de l'entrée avec tous les exemples
    entry = {
        'word': word,
        'level': get_word_level(word),
        'translations': translations,
        'examples': examples,  # Prend tous les exemples générés
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
    """Génère des exercices d'apprentissage pour un mot allemand
    
    Args:
        word: Le mot pour lequel générer des exercices
        models: Les modèles de traduction et génération
        difficulty: Niveau de difficulté ('beginner', 'intermediate', 'advanced')
        
    Returns:
        dict: Dictionnaire contenant les différents types d'exercices
    """
    tokenizer_gpt, model_gpt = models['gpt']
    tokenizer_de_fr, model_de_fr = models['de_fr']
    exercises = {}
    
    clean_word_input = clean_word(word)
    
    try:
        # 1. Exercice à trous - génération de phrases contextuelles
        prompt = f"""Schreiben Sie drei verschiedene Sätze mit dem Wort '{clean_word_input}':
1. Ein sehr einfacher Satz für Anfänger (mit Alltagsvokabular).
2. Ein Satz mittlerer Schwierigkeit mit einer wichtigen Redewendung.
3. Ein anspruchsvoller Satz mit komplexerer Grammatik."""

        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=5,
            temperature=0.7,
            do_sample=True
        )
        sentences_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).split('\n')
        sentences_de = [s.strip() for s in sentences_de if s.strip() and clean_word_input in s.lower()]

        fill_blank_exercises = []
        for i, sent_de in enumerate(sentences_de[:3]):
            # Traduction en français
            inputs = tokenizer_de_fr(sent_de, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            
            # Reformulation en français naturel
            translation = reformat_french_text(translation, models, context='example')
            
            fill_blank_exercises.append({
                'de': sent_de.replace(clean_word_input, '___'),
                'fr': translation,
                'answer': clean_word_input,
                'difficulty': ['beginner', 'intermediate', 'advanced'][i] if i < 3 else 'advanced'
            })

        exercises['fill_blank'] = {'de': fill_blank_exercises}

        # 2. Choix multiples avec contexte réel
        prompt = f"""Erstellen Sie eine Multiple-Choice-Frage für das Wort '{clean_word_input}':
Frage: Wie wird das Wort '{clean_word_input}' korrekt verwendet?
A) [Korrekte Verwendung mit typischem Kontext]
B) [Ähnliche, aber leicht falsche Verwendung]
C) [Häufiger Fehler bei der Verwendung]
D) [Komplett falsche Verwendung]
Richtige Antwort: A"""

        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=5,
            temperature=0.6
        )
        qcm_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Traduction et reformulation du QCM
        inputs = tokenizer_de_fr(qcm_de, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(**inputs)
        qcm_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)

        # Extrait la question et les choix
        de_lines = qcm_de.split('\n')
        fr_lines = qcm_fr.split('\n')

        question_de = next((l for l in de_lines if '?' in l), '')
        question_fr = reformat_french_text(next((l for l in fr_lines if '?' in l), ''), 
                                         models, context='exercise_question')

        choices_de = [l[3:].strip() for l in de_lines if l.startswith(('A)', 'B)', 'C)', 'D)'))]
        choices_fr = [reformat_french_text(l[3:].strip(), models, context='exercise_choice') 
                     for l in fr_lines if l.startswith(('A)', 'B)', 'C)', 'D)'))]

        exercises['multiple_choice'] = {
            'de': {
                'question': question_de,
                'choices': choices_de,
                'answer': 'A'
            },
            'fr': {
                'question': question_fr,
                'choices': choices_fr,
                'answer': 'A'
            }
        }

        # 3. Association de mots avec contexte
        prompt = f"""Geben Sie 6 wichtige verwandte Wörter zu '{clean_word_input}' an:
1. Ein gebräuchliches Synonym
2. Ein häufig verwendetes Verb in diesem Kontext
3. Ein passendes Adjektiv
4. Ein klarer Gegenbegriff
5. Die übergeordnete Kategorie
6. Ein spezifischerer Begriff"""

        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=5,
            temperature=0.6
        )
        
        related_words_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).split('\n')
        related_words_de = [w.split('.')[-1].strip() if '.' in w else w.strip() 
                          for w in related_words_de if w.strip()]

        # Traduction et reformulation des mots associés
        related_words_fr = []
        for word_de in related_words_de:
            inputs = tokenizer_de_fr(word_de, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            reformulated = reformat_french_text(translation, models, context='definition')
            related_words_fr.append(reformulated)

        categories_de = ['Synonym', 'Verb', 'Adjektiv', 'Antonym', 
                        'Oberbegriff', 'Unterbegriff']
        categories_fr = ['Synonyme', 'Verbe associé', 'Adjectif', 'Antonyme', 
                        'Catégorie', 'Terme spécifique']

        exercises['word_association'] = {
            'de': {
                'words': related_words_de,
                'categories': categories_de
            },
            'fr': {
                'words': related_words_fr,
                'categories': categories_fr
            }
        }
        
        return exercises
    except Exception as e:
        print(f"Erreur lors de la génération des exercices: {e}")
        return None

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

def reformat_french_text(text, models, context=None):
    """Reformule un texte français pour le rendre plus naturel et pédagogique
    
    Args:
        text: Le texte français à reformuler
        models: Les modèles de traduction
        context: Le contexte d'utilisation (exercice, définition, exemple, etc.)
        
    Returns:
        str: Le texte reformulé
    """
    tokenizer_gpt, model_gpt = models['gpt']
    
    try:
        # Adapter le prompt selon le contexte
        if context == 'exercise_question':
            prompt = f"""Réécrivez cette consigne d'exercice en français clair et pédagogique, 
en gardant l'aspect didactique : '{text}'"""
        elif context == 'exercise_choice':
            prompt = f"""Reformulez cette option de réponse en français naturel 
et pertinent pour un exercice de langue : '{text}'"""
        elif context == 'definition':
            prompt = f"""Donnez une définition claire et précise en français pour : '{text}'
La définition doit être concise mais complète."""
        elif context == 'example':
            prompt = f"""Reformulez cet exemple en une phrase naturelle en français : '{text}'
La phrase doit être grammaticalement correcte et avoir du sens."""
        else:
            prompt = f"""Reformulez ce texte en français naturel et élégant : '{text}'
Assurez-vous que la formulation soit claire et correcte grammaticalement."""

        inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=len(text) + 100,  # Plus de marge pour une bonne reformulation
            num_beams=5,
            temperature=0.4,  # Un peu plus de créativité tout en restant fidèle
            top_p=0.9,
            do_sample=True,  # Permettre un peu de variation
            no_repeat_ngram_size=3  # Éviter les répétitions
        )
        reformulated = model_gpt.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyage et amélioration du texte reformulé
        reformulated = reformulated.strip()
        
        # Retirer les guillemets si présents
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1].strip()
            
        # Assurer que la phrase se termine par un point si c'est une phrase complète
        if any(reformulated[0].isupper() for c in reformulated) and not reformulated.endswith(('.', '?', '!')):
            reformulated += '.'
            
        # Vérifier que le texte n'est pas vide ou trop court
        if len(reformulated) < 3:
            return text
            
        # Vérifier que le texte reformulé garde le sens original
        if len(reformulated.split()) < len(text.split()) / 2:
            return text  # La reformulation est trop courte, garder l'original
            
        return reformulated
    except Exception as e:
        print(f"Erreur lors de la reformulation: {e}")
        return text
