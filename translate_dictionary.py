#!/usr/bin/env python3
import sqlite3
import csv
import re
from transformers import MarianMTModel, MarianTokenizer
import os
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from wiktionaryparser import WiktionaryParser
import time

# Configuration
DB_PATH = "dictionary.sqlite"
INPUT_FILE = "de-en.txt"
MODEL_PATH = "./opus-mt-en-fr"  # Modèle pour traduire de l'anglais vers le français

def setup_database():
    """Création de la base de données et de la table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Création de la table dictionary
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            de TEXT NOT NULL,
            en TEXT NOT NULL,
            fr TEXT NOT NULL,
            wordType TEXT,
            wiktionary_def TEXT,
            dictcc_def TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Création des index pour optimiser les recherches
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_de ON dictionary(de)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_en ON dictionary(en)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fr ON dictionary(fr)")
    
    conn.commit()
    return conn

def load_translation_model():
    """Chargement du modèle de traduction"""
    print("Chargement du modèle de traduction EN-FR...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
    model = MarianMTModel.from_pretrained(MODEL_PATH)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    """Traduction d'un texte de l'anglais vers le français"""
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

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
    curly_regex = r'{\K[^}]*(?=})'
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

def get_wiktionary_definition(word):
    """Récupère la définition depuis Wiktionary"""
    parser = WiktionaryParser()
    try:
        result = parser.fetch(word, 'german')
        if result and result[0].get('definitions'):
            return result[0]['definitions'][0].get('text', [''])[0]
    except Exception as e:
        print(f"Erreur Wiktionary pour {word}: {e}")
    return None

def get_dictcc_definition(word):
    """Récupère la définition depuis Dict.cc"""
    url = f"https://www.dict.cc/?s={word}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        definition = soup.find('td', {'class': 'td7nl'})
        return definition.text.strip() if definition else None
    except Exception as e:
        print(f"Erreur Dict.cc pour {word}: {e}")
    return None

def main():
    # Initialisation de la base de données
    conn = setup_database()
    cursor = conn.cursor()
    
    # Chargement du modèle de traduction
    tokenizer, model = load_translation_model()
    
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
                fr_translation = translate_text(result['en'], tokenizer, model)
                
                # Récupération des définitions (avec délai pour éviter le rate limiting)
                print(f"\n{'='*50}")
                print(f"Données récupérées pour le mot : {result['de']}")
                print(f"{'='*50}")
                print(f"Allemand     : {result['de']}")
                print(f"Type de mot  : {result['wordType']}")
                print(f"Anglais      : {result['en']}")
                print(f"Français     : {fr_translation}")
                
                wiktionary_def = get_wiktionary_definition(result['de'])
                time.sleep(1)  # Délai de 1 seconde entre les requêtes
                print(f"Wiktionary   : {wiktionary_def}")
                
                dictcc_def = get_dictcc_definition(result['de'])
                time.sleep(1)  # Délai de 1 seconde entre les requêtes
                print(f"Dict.cc      : {dictcc_def}")
                print(f"{'='*50}\n")
                
                # Insertion dans la base de données
                cursor.execute("""
                    INSERT INTO dictionary (de, en, fr, wordType, wiktionary_def, dictcc_def)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result['de'],
                    result['en'],
                    fr_translation,
                    result['wordType'],
                    wiktionary_def,
                    dictcc_def
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
    cursor.execute("SELECT de, en, fr, wordType, wiktionary_def, dictcc_def FROM dictionary LIMIT 5")
    for row in cursor.fetchall():
        print(f"DE: {row[0]}")
        print(f"EN: {row[1]}")
        print(f"FR: {row[2]}")
        print(f"Type: {row[3]}")
        if row[4]:  # wiktionary_def
            print(f"Wiktionary: {row[4]}")
        if row[5]:  # dictcc_def
            print(f"Dict.cc: {row[5]}")
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
