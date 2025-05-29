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
    examples = []        # Prompts pour des phrases courtes et complètes
    prompts = [
        f"Schreiben Sie einen kurzen, vollständigen Satz mit '{word}' (maximal 8 Wörter).",
        f"Bilden Sie einen einfachen Satz mit '{word}' (maximal 6 Wörter).",
        f"Machen Sie einen präzisen Satz mit '{word}' (maximal 7 Wörter).",
        f"Erstellen Sie einen klaren Satz mit '{word}' (maximal 5 Wörter)."
    ]
    
    for prompt in prompts:
        try:
            # Génération en allemand
            inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True)
            outputs = model_gpt.generate(
                inputs.input_ids,
                max_length=30,   # Assez long pour une phrase complète
                min_length=5,    # Assez court pour rester concis
                num_beams=5,     # Plus de beams pour de meilleures phrases
                temperature=0.4,  # Un peu plus de créativité
                no_repeat_ngram_size=2,  # Éviter les répétitions
                do_sample=True,  # Permettre un peu de variation
                top_p=0.9,       # Garder les meilleures options
                early_stopping=True  # Arrêter quand la phrase est complète
            )
            german = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Nettoyer et valider la phrase allemande
            if not any(german.endswith(p) for p in ['.', '!', '?']):
                german += '.'
            
            # Vérifier que la phrase est complète et contient le mot
            if not german or word.lower() not in german.lower() or len(german.split()) > 8:
                continue  # Ignorer les phrases invalides
            
            # Traduction en français
            inputs = tokenizer_de_fr(german, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(**inputs)
            french = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Nettoyer et valider la traduction française
            if not any(french.endswith(p) for p in ['.', '!', '?']):
                french += '.'
            
            # Vérifier que la phrase française est complète
            if len(french.split()) <= 2 or len(french.split()) > 10:
                continue  # Ignorer les traductions trop courtes ou trop longues
                
            # Reformuler pour plus de naturel
            french = reformat_french_text(french, models, context='example')
            
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
        # 1. Obtenir et valider la définition en allemand
        prompt_def_de = f"Definieren Sie das Wort '{word}' in einem klaren und präzisen Satz:"
        inputs = tokenizer_gpt(prompt_def_de, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=3,
            temperature=0.3,
            do_sample=False
        )
        definition_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).strip()
        if definition_de and len(definition_de.split()) >= 3:  # Validation minimale
            translations['de']['definition'] = definition_de

        # 2. Obtenir et valider les synonymes en allemand
        prompt_syn_de = f"Nennen Sie bis zu 5 genaue Synonyme für '{word}' (ein Wort pro Zeile):"
        inputs = tokenizer_gpt(prompt_syn_de, return_tensors="pt", padding=True)
        outputs = model_gpt.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=5,
            temperature=0.7,
            do_sample=True
        )
        synonyms_de = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
        # Filtrer les synonymes valides
        translations['de']['synonymes'] = [
            s.strip() for s in synonyms_de.split('\n')
            if s.strip() and s.strip().lower() != word.lower()
        ][:5]

        # 3. Traduction principale en français avec validation
        inputs = tokenizer_de_fr(word, return_tensors="pt", padding=True)
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=20,
            num_beams=5,  # Augmenté pour une meilleure qualité
            temperature=0.3,
            do_sample=False
        )
        translated = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
        formatted_translation = reformat_french_text(translated, models, context='definition')
        if formatted_translation:
            translations['fr']['principal'] = formatted_translation

        # 4. Traduction de la définition allemande en français
        if translations['de']['definition']:
            inputs = tokenizer_de_fr(translations['de']['definition'], return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(
                inputs.input_ids,
                max_length=100,  # Augmenté pour les définitions plus longues
                num_beams=5,
                temperature=0.3,
                do_sample=False
            )
            translated_def = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
            formatted_def = reformat_french_text(translated_def, models, context='definition')
            if formatted_def:
                translations['fr']['definition'] = formatted_def

        # 5. Générer des variantes de traduction en français
        seen_translations = {translations['fr']['principal']}
        outputs = model_de_fr.generate(
            inputs.input_ids,
            max_length=20,
            num_return_sequences=5,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            diversity_penalty=0.8  # Augmenté pour plus de diversité
        )
        
        for output in outputs:
            translation = tokenizer_de_fr.decode(output, skip_special_tokens=True)
            reformulated = reformat_french_text(translation, models)
            if reformulated and reformulated not in seen_translations and len(reformulated.split()) >= 1:
                translations['fr']['variantes'].append(reformulated)
                seen_translations.add(reformulated)

        # 6. Traduire les synonymes allemands en français
        translations['fr']['synonymes'] = []
        for syn in translations['de']['synonymes']:
            try:
                inputs = tokenizer_de_fr(syn, return_tensors="pt", padding=True)
                outputs = model_de_fr.generate(
                    inputs.input_ids,
                    max_length=20,
                    num_beams=5,
                    temperature=0.3,
                    do_sample=False
                )
                syn_fr = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True)
                formatted_syn = reformat_french_text(syn_fr, models)
                if formatted_syn and formatted_syn not in translations['fr']['synonymes']:
                    translations['fr']['synonymes'].append(formatted_syn)
            except Exception:
                continue

    except Exception as e:
        print(f"Erreur lors de la traduction de '{word}': {e}")
        if not translations['fr']['principal']:
            translations['fr']['principal'] = word

    # Validation finale et nettoyage
    translations['fr']['variantes'] = list(dict.fromkeys(translations['fr']['variantes']))[:5]
    translations['fr']['synonymes'] = list(dict.fromkeys(translations['fr']['synonymes']))[:5]
    
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
        # 1. Exercice à trous - génération de phrases contextuelles simples
        fill_blank_exercises = []
        
        # Templates de phrases simples adaptés au niveau
        sentence_templates = {
            'beginner': [
                f"Ich __ {clean_word_input}.",
                f"Das ist ein __ {clean_word_input}.",
                f"Wir haben __ {clean_word_input}."
            ],
            'intermediate': [
                f"Gestern habe ich __ {clean_word_input} gekauft.",
                f"Der __ {clean_word_input} ist sehr schön.",
                f"Kannst du mir den __ {clean_word_input} geben?"
            ],
            'advanced': [
                f"Obwohl der __ {clean_word_input} teuer war, haben wir ihn gekauft.",
                f"Nachdem ich den __ {clean_word_input} gesehen hatte, war ich beeindruckt.",
                f"Je mehr ich über __ {clean_word_input} lerne, desto interessanter wird es."
            ]
        }
        
        # Générer des phrases naturelles pour chaque template
        for i, template in enumerate(sentence_templates.get(difficulty, sentence_templates['intermediate'])[:3]):
            try:
                # Template simple pour démarrer la génération
                prompt = f"Vervollständigen Sie diesen Satz auf natürliche Weise: {template}"
                
                inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)
                outputs = model_gpt.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 30,  # Longueur raisonnable
                    num_beams=3,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=tokenizer_gpt.eos_token_id
                )
                
                generated_sentence = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
                # Nettoyer la phrase générée
                generated_sentence = generated_sentence.replace(prompt, "").strip()
                
                # Si la génération n'est pas satisfaisante, utiliser un exemple prédéfini
                if not generated_sentence or len(generated_sentence) < 5:
                    predefined_sentences = {
                        'beginner': [
                            f"Ich mag {clean_word_input}.",
                            f"Das ist mein {clean_word_input}.",
                            f"Wir brauchen {clean_word_input}."
                        ],
                        'intermediate': [
                            f"Gestern habe ich einen {clean_word_input} gekauft.",
                            f"Der neue {clean_word_input} funktioniert gut.",
                            f"Kannst du mir deinen {clean_word_input} leihen?"
                        ],
                        'advanced': [
                            f"Obwohl der {clean_word_input} kompliziert ist, verstehe ich ihn.",
                            f"Nachdem ich den {clean_word_input} studiert hatte, war alles klarer.",
                            f"Je öfter ich {clean_word_input} benutze, desto besser wird es."
                        ]
                    }
                    generated_sentence = predefined_sentences[difficulty][i]
                
                # Traduction de la phrase
                inputs_fr = tokenizer_de_fr(generated_sentence, return_tensors="pt", padding=True)
                outputs_fr = model_de_fr.generate(
                    inputs_fr.input_ids,
                    max_length=inputs_fr.input_ids.shape[1] + 20,
                    num_beams=3,
                    temperature=0.3
                )
                sentence_fr = tokenizer_de_fr.decode(outputs_fr[0], skip_special_tokens=True)
                
                # Créer l'exercice à trous
                sentence_with_blank = generated_sentence.replace(clean_word_input, "____")
                sentence_fr_with_blank = sentence_fr.replace(clean_word_input.lower(), "____")
                
                fill_blank_exercises.append({
                    'de': {
                        'text': sentence_with_blank,
                        'answer': clean_word_input,
                        'complete_sentence': generated_sentence
                    },
                    'fr': {
                        'text': sentence_fr_with_blank,
                        'translation': sentence_fr,
                        'instruction': f"Complétez avec le mot allemand pour '{clean_word_input}'"
                    }
                })
                
            except Exception as e:
                print(f"Erreur pour l'exercice à trous {i+1}: {e}")
                continue
        
        exercises['fill_blank'] = fill_blank_exercises

        # 2. Choix multiples - créer des questions sur le sens et l'usage
        try:
            # Obtenir d'abord la traduction principale
            inputs = tokenizer_de_fr(clean_word_input, return_tensors="pt", padding=True)
            outputs = model_de_fr.generate(inputs.input_ids, max_length=20, num_beams=3)
            main_translation = tokenizer_de_fr.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Créer des choix plausibles mais incorrects
            distractors = []
            similar_words = ['maison', 'voiture', 'livre', 'chien', 'eau', 'temps', 'personne', 'travail']
            for word_fr in similar_words:
                if word_fr != main_translation.lower():
                    distractors.append(word_fr)
                if len(distractors) >= 3:
                    break
            
            choices = [main_translation] + distractors[:3]
            import random
            random.shuffle(choices)
            correct_index = choices.index(main_translation)
            correct_letter = ['A', 'B', 'C', 'D'][correct_index]
            
            exercises['multiple_choice'] = {
                'de': {
                    'question': f"Was bedeutet das Wort '{clean_word_input}' auf Französisch?",
                    'choices': [f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)],
                    'answer': correct_letter,
                    'explanation': f"'{clean_word_input}' bedeutet '{main_translation}' auf Französisch."
                },
                'fr': {
                    'question': f"Que signifie le mot allemand '{clean_word_input}' ?",
                    'choices': [f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)],
                    'answer': correct_letter,
                    'explanation': f"'{clean_word_input}' signifie '{main_translation}' en français."
                }
            }
            
        except Exception as e:
            print(f"Erreur pour le QCM: {e}")
            exercises['multiple_choice'] = None

        # 3. Association de mots - générer des mots sémantiquement liés
        try:
            # Créer des associations basées sur des catégories sémantiques
            word_associations = {
                'de': {
                    'target_word': clean_word_input,
                    'associated_words': [],
                    'categories': ['Synonym', 'Gegenteil', 'Verwandt', 'Bereich']
                },
                'fr': {
                    'target_word': main_translation,
                    'associated_words': [],
                    'categories': ['Synonyme', 'Contraire', 'Lié', 'Domaine']
                }
            }
            
            # Générer des mots associés simples
            association_prompts = [
                f"Ein Wort ähnlich wie {clean_word_input}:",
                f"Das Gegenteil von {clean_word_input}:",
                f"Ein Wort aus demselben Bereich wie {clean_word_input}:",
                f"Ein verwandter Begriff zu {clean_word_input}:"
            ]
            
            for prompt in association_prompts[:4]:  # Limiter à 4 associations
                try:
                    inputs = tokenizer_gpt(prompt, return_tensors="pt", padding=True, truncation=True)
                    outputs = model_gpt.generate(
                        inputs.input_ids,
                        max_length=inputs.input_ids.shape[1] + 10,
                        num_beams=3,
                        temperature=0.6,
                        pad_token_id=tokenizer_gpt.eos_token_id
                    )
                    associated_word = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)
                    associated_word = associated_word.replace(prompt, "").strip().split()[0]
                    
                    # Traduction du mot associé
                    inputs_fr = tokenizer_de_fr(associated_word, return_tensors="pt", padding=True)
                    outputs_fr = model_de_fr.generate(inputs_fr.input_ids, max_length=15)
                    associated_word_fr = tokenizer_de_fr.decode(outputs_fr[0], skip_special_tokens=True).strip()
                    
                    word_associations['de']['associated_words'].append(associated_word)
                    word_associations['fr']['associated_words'].append(associated_word_fr)
                    
                except Exception as e:
                    print(f"Erreur pour l'association: {e}")
                    continue
            
            exercises['word_association'] = word_associations
            
        except Exception as e:
            print(f"Erreur pour les associations de mots: {e}")
            exercises['word_association'] = None

        # 4. Exercice de construction de phrases
        try:
            sentence_building = {
                'de': {
                    'instruction': f"Bilden Sie einen Satz mit dem Wort '{clean_word_input}'",
                    'example': f"Beispiel: Ich verwende {clean_word_input} jeden Tag.",
                    'level': difficulty
                },
                'fr': {
                    'instruction': f"Construisez une phrase avec le mot allemand '{clean_word_input}'",
                    'example': f"Exemple: J'utilise {clean_word_input} tous les jours.",
                    'level': difficulty
                }
            }
            exercises['sentence_building'] = sentence_building
            
        except Exception as e:
            print(f"Erreur pour la construction de phrases: {e}")
            exercises['sentence_building'] = None
        
    except Exception as e:
        print(f"Erreur générale lors de la génération des exercices: {e}")
        return {
            'error': str(e),
            'word': clean_word_input,
            'fallback_exercises': {
                'fill_blank': [{
                    'de': {'text': f"Ich benutze ____.", 'answer': clean_word_input},
                    'fr': {'text': f"J'utilise ____.", 'instruction': "Complétez en allemand"}
                }]
            }
        }
    
    return exercises



def format_exercise_output(exercises, word):
    """Formate la sortie des exercices pour un affichage cohérent"""
    formatted = {
        'word': word,
        'total_exercises': 0,
        'exercise_types': []
    }
    
    for exercise_type, exercise_data in exercises.items():
        if exercise_data and exercise_type != 'error':
            formatted['exercise_types'].append(exercise_type)
            if exercise_type == 'fill_blank' and isinstance(exercise_data, list):
                formatted['total_exercises'] += len(exercise_data)
            else:
                formatted['total_exercises'] += 1
    
    formatted['exercises'] = exercises
    return formatted
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
