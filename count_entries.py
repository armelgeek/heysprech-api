#!/usr/bin/env python3
import sqlite3
import json

# Configuration
DB_PATH = "dictionary.sqlite"

def count_entries():
    """Affiche les statistiques de la base de données"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Nombre total d'entrées
    cursor.execute("SELECT COUNT(*) FROM dictionary")
    total = cursor.fetchone()[0]
    print(f"\nNombre total d'entrées: {total}")
    
    # Nombre d'entrées par type de mot
    cursor.execute("SELECT wordType, COUNT(*) FROM dictionary GROUP BY wordType")
    print("\nRépartition par type de mot:")
    for word_type, count in cursor.fetchall():
        print(f"- {word_type}: {count} mots")
    
    # Nombre d'entrées avec exemples
    cursor.execute("SELECT COUNT(*) FROM dictionary WHERE example_sentences IS NOT NULL")
    with_examples = cursor.fetchone()[0]
    print(f"\nEntrées avec phrases d'exemple: {with_examples}")
    
    # Nombre d'entrées avec définitions
    cursor.execute("SELECT COUNT(*) FROM dictionary WHERE wiktionary_def IS NOT NULL")
    with_wiktionary = cursor.fetchone()[0]
    print(f"Entrées avec définition Wiktionary: {with_wiktionary}")
    
    cursor.execute("SELECT COUNT(*) FROM dictionary WHERE dictcc_def IS NOT NULL")
    with_dictcc = cursor.fetchone()[0]
    print(f"Entrées avec définition Dict.cc: {with_dictcc}")
    
    # Quelques exemples récents
    print("\nDernières entrées ajoutées:")
    cursor.execute("""
        SELECT de, en, fr, wordType, example_sentences 
        FROM dictionary 
        ORDER BY created_at DESC 
        LIMIT 3
    """)
    
    for row in cursor.fetchall():
        de, en, fr, word_type, examples = row
        print(f"\n{'-'*50}")
        print(f"Mot: {de}")
        print(f"Type: {word_type}")
        print(f"EN: {en}")
        print(f"FR: {fr}")
        if examples:
            examples_list = json.loads(examples)
            print("\nExemples:")
            for i, example in enumerate(examples_list, 1):
                print(f"  {i}. DE: {example['de']}")
                print(f"     EN: {example['en']}")
                print(f"     FR: {example['fr']}")
    
    conn.close()

if __name__ == "__main__":
    count_entries()
