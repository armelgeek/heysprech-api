#!/usr/bin/env python3
import sqlite3
import json

# Configuration
DB_PATH = "dictionary.sqlite"

def count_entries():
    """Affiche les statistiques de traduction allemand-français"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Nombre total d'entrées
    cursor.execute("SELECT COUNT(*) FROM dictionary")
    total = cursor.fetchone()[0]
    print(f"\n{'='*50}")
    print(f"BASE DE DONNÉES DE TRADUCTION ALLEMAND-FRANÇAIS")
    print(f"{'='*50}")
    print(f"Nombre total de mots : {total}")
    
    # Nombre d'entrées par type de mot
    print(f"\n{'='*50}")
    print("RÉPARTITION PAR TYPE DE MOT")
    print(f"{'='*50}")
    cursor.execute("SELECT wordType, COUNT(*) FROM dictionary GROUP BY wordType")
    for word_type, count in cursor.fetchall():
        print(f"{word_type:<20}: {count} mots")
    
    # Dernières traductions ajoutées
    print(f"\n{'='*50}")
    print("DERNIÈRES TRADUCTIONS AJOUTÉES")
    print(f"{'='*50}")
    cursor.execute("""
        SELECT de, fr, wordType, created_at 
        FROM dictionary 
        ORDER BY created_at DESC 
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        print(f"\nMot allemand : {row[0]}")
        print(f"Traduction   : {row[1]}")
        print(f"Type de mot  : {row[2]}")
        print(f"Ajouté le    : {row[3]}")
        print("-" * 40)
    
    conn.close()

if __name__ == "__main__":
    count_entries()
