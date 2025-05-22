#!/usr/bin/env python3
import sqlite3
import csv
import re
from transformers import MarianMTModel, MarianTokenizer
import os
from tqdm import tqdm
import time

# Configuration
DB_PATH = "dictionary.sqlite"
INPUT_FILE = "de-en.txt"
MODEL_PATH_EN_FR = "./opus-mt-en-fr"  # Modèle pour traduire de l'anglais vers le français
MODEL_PATH_DE_EN = "./opus-mt-de-en"  # Modèle pour traduire de l'allemand vers l'anglais

def setup_database():
    """Création de la base de données et de la table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Création de la table dictionary
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            de TEXT NOT NULL,
            fr TEXT NOT NULL,
            wordType TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Création des index pour optimiser les recherches
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_de ON dictionary(de)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fr ON dictionary(fr)")
    
    conn.commit()
    return conn

def load_translation_models():
    """Chargement des modèles de traduction"""
    print("Chargement du modèle de traduction DE-EN...")
    tokenizer_de_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model_de_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    
    print("Chargement du modèle de traduction EN-FR...")
    tokenizer_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    
    return {
        'de_en': (tokenizer_de_en, model_de_en),
        'en_fr': (tokenizer_en_fr, model_en_fr)
    }

def translate_text(text, models, source_lang='en'):
    """Traduction d'un texte vers le français
    
    Args:
        text: Texte à traduire
        models: Dictionnaire contenant les modèles de traduction
        source_lang: Langue source ('de' pour allemand, 'en' pour anglais)
    """
    if source_lang == 'de':
        # Traduction allemand -> anglais -> français
        tokenizer_de_en, model_de_en = models['de_en']
        tokenizer_en_fr, model_en_fr = models['en_fr']
        
        # Première traduction : DE -> EN
        inputs_de = tokenizer_de_en(text, return_tensors="pt", padding=True)
        outputs_de = model_de_en.generate(**inputs_de)
        english = tokenizer_de_en.decode(outputs_de[0], skip_special_tokens=True)
        
        # Deuxième traduction : EN -> FR
        inputs_en = tokenizer_en_fr(english, return_tensors="pt", padding=True)
        outputs_en = model_en_fr.generate(**inputs_en)
        return tokenizer_en_fr.decode(outputs_en[0], skip_special_tokens=True)
    else:
        # Traduction directe anglais -> français
        tokenizer_en_fr, model_en_fr = models['en_fr']
        inputs = tokenizer_en_fr(text, return_tensors="pt", padding=True)
        outputs = model_en_fr.generate(**inputs)
        return tokenizer_en_fr.decode(outputs[0], skip_special_tokens=True)

def parse_dictionary_line(line):
    """Parse une ligne du dictionnaire"""
    # Séparation allemand et anglais
    try:
        de_part, en_part = line.split("::")
    except ValueError:
        return None
    
    # Traitement de la partie allemande
    de_array = de_part.split(";")[0]  # Prendre la première définition
    de_word = de_array.split("|")[0]  # Prendre la première forme
    
    # Extraction du type de mot
    curly_regex = r'{([^}]*)}'  # Capture le contenu entre accolades
    type_matches = re.findall(curly_regex, de_word)
    word_type = type_matches[0] if type_matches else None
    
    # Nettoyage du mot allemand
    clean_de = re.match(r'[^{]*', de_word).group(0).strip()
    
    # Suppression des articles allemands
    clean_de = re.sub(r'^(das|der|die) ', '', clean_de, flags=re.IGNORECASE)
    
    # Traitement de la partie anglaise
    en_word = en_part.split("|")[0]  # Prendre la première traduction
    en_word = en_word.split(";")[0].strip()  # Prendre la première définition
    
    return {
        'de': clean_de,
        'en': en_word,
        'wordType': word_type
    }

def count_lines(filename):
    """Compte le nombre de lignes dans le fichier"""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)
    """Génère et traduit trois phrases d'exemple pour un mot donné utilisant un modèle T5"""
    examples = []
    
    try:
        # Initialisation du modèle T5 pour l'allemand si pas déjà fait
        if 'text_generator' not in models:
            from transformers import GPT2LMHeadModel, AutoTokenizer
            print("Chargement du modèle de génération de texte allemand...")
            model_name = "benjamin/gpt2-wechsel-german"
            print(f"Chargement du modèle {model_name}...")
            tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
            tokenizer_gpt2.padding_side = 'left'
            models['text_generator'] = (tokenizer_gpt2, model_gpt2)
        
        tokenizer_t5, model_t5 = models['text_generator']
        
        # Prompts selon le type de mot
        if word_type and word_type.lower().startswith('v'):
            prompts = [
                f"Verwende das Verb '{word}' in einem Satz: ",
                f"Schreibe einen Satz mit dem Verb '{word}' in der Vergangenheit: ",
                f"Beschreibe mit dem Verb '{word}' eine zukünftige Aktion: "
            ]
        elif word_type and word_type.lower().startswith('adj'):
            prompts = [
                f"Beschreibe etwas mit dem Adjektiv '{word}': ",
                f"Verwende '{word}' um eine Person zu beschreiben: ",
                f"Beschreibe eine Situation mit dem Wort '{word}': "
            ]
        else:  # Noms par défaut
            prompts = [
                f"Beschreibe das '{word}': ",
                f"Schreibe einen Satz über ein '{word}': ",
                f"Verwende '{word}' in einem interessanten Kontext: "
            ]
    
        # Configuration des modèles de traduction
        tokenizer_de_en, model_de_en = models['de_en']
        tokenizer_en_fr, model_en_fr = models['en_fr']
        
        # Génération et traduction des phrases
        for prompt in prompts:
            # Génération de la phrase en allemand
            inputs = tokenizer_t5(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model_t5.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True
            )
            german_sentence = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
            
            # Traduction en anglais
            inputs_de = tokenizer_de_en(german_sentence, return_tensors="pt", padding=True)
            outputs_de = model_de_en.generate(**inputs_de)
            english = tokenizer_de_en.decode(outputs_de[0], skip_special_tokens=True)
            
            # Traduction en français
            inputs_en = tokenizer_en_fr(english, return_tensors="pt", padding=True)
            outputs_en = model_en_fr.generate(**inputs_en)
            french = tokenizer_en_fr.decode(outputs_en[0], skip_special_tokens=True)
            
            examples.append({
                'de': german_sentence,
                'en': english,
                'fr': french
            })
        
        return examples
        
    except Exception as e:
        print(f"Erreur lors de la génération de phrases: {str(e)}")
        return []  # Retourner une liste vide en cas d'erreur

def main():
    # Initialisation de la base de données
    conn = setup_database()
    cursor = conn.cursor()
    
    # Chargement des modèles de traduction
    models = load_translation_models()
    
    print(f"Traitement du fichier {INPUT_FILE}...")
    total_lines = count_lines(INPUT_FILE)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        pbar = tqdm(csv_reader, total=total_lines, desc="Traitement des mots")
        
        for row in pbar:
            if not row:
                continue
                
            # Parse la ligne
            result = parse_dictionary_line(row[0])
            if not result or not result['wordType']:
                continue
            
            # Vérifier si le mot existe déjà
            cursor.execute("SELECT id FROM dictionary WHERE de = ?", (result['de'],))
            if cursor.fetchone():
                continue
            
            try:
                # Traduction en français
                fr_translation = translate_text(result['de'], models, source_lang='de')
                
                # Affichage des informations
                print(f"\n{'='*50}")
                print(f"Traduction du mot : {result['de']}")
                print(f"{'='*50}")
                print(f"Allemand     : {result['de']}")
                print(f"Type de mot  : {result['wordType']}")
                print(f"Français     : {fr_translation}")
                print(f"{'='*50}\n")
                
                cursor.execute("""
                    INSERT INTO dictionary (de, fr, wordType)
                    VALUES (?, ?, ?)
                """, (
                    result['de'],
                    fr_translation,
                    result['wordType']
                ))
                
                # Commit toutes les 100 insertions
                if cursor.lastrowid % 100 == 0:
                    conn.commit()
                    
                # Mise à jour de la description de la barre de progression
                pbar.set_postfix({"Dernier mot": result['de']})
                
            except Exception as e:
                print(f"\nErreur lors du traitement de {result['de']}: {str(e)}")
                continue
    
    # Commit final
    conn.commit()
    
    # Afficher les statistiques
    cursor.execute("SELECT COUNT(*) FROM dictionary")
    total_words = cursor.fetchone()[0]
    print(f"\nTerminé! {total_words} mots stockés dans {DB_PATH}")
    
    # Quelques exemples de traductions
    print("\nExemples de traductions:")
    cursor.execute("SELECT de, fr, wordType FROM dictionary LIMIT 5")
    for row in cursor.fetchall():
        print(f"DE: {row[0]}")
        print(f"FR: {row[1]}")
        print(f"Type: {row[2]}")
        print("-" * 40)
    
    # Fermeture de la connexion
    conn.close()
    
    # Création du fichier d'indication de fin de traitement
    completion_time = time.strftime("%Y-%m-%d %H:%M:%S")
    with open("dictionary_processing_complete.txt", "w", encoding="utf-8") as f:
        f.write(f"Traitement du dictionnaire terminé le {completion_time}\n")
        f.write(f"Nombre total de mots traités : {total_words}\n")
        f.write("Toutes les opérations ont été effectuées avec succès.")
    
    print(f"\nFichier de confirmation créé : dictionary_processing_complete.txt")

if __name__ == "__main__":
    main()
