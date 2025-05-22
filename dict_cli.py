#!/usr/bin/env python3
import click
import sqlite3
import textwrap
from pathlib import Path

DB_PATH = "dictionary.sqlite"

def format_word_info(word_info, show_all=False):
    """Formate les informations d'un mot pour l'affichage"""
    output = []
    output.append("=" * 50)
    output.append(f"🇩🇪  {word_info[0]}")
    output.append(f"🇬🇧  {word_info[1]}")
    output.append(f"🇫🇷  {word_info[2]}")
    output.append(f"Type: {word_info[3]}")
    
    if show_all and len(word_info) > 4:
        if word_info[4]:  # wiktionary_def
            output.append("\nDéfinition Wiktionary:")
            output.append(textwrap.indent(word_info[4], "  "))
        
        if word_info[5]:  # dictcc_def
            output.append("\nDéfinition Dict.cc:")
            output.append(textwrap.indent(word_info[5], "  "))
            
        if word_info[6] and word_info[7]:  # examples
            output.append("\nExemple:")
            output.append(f"  DE: {word_info[6]}")
            output.append(f"  FR: {word_info[7]}")
            
    return "\n".join(output)

@click.group()
def cli():
    """Dictionnaire Allemand-Français en ligne de commande"""
    pass

@cli.command()
@click.argument('word')
@click.option('--lang', default='de', type=click.Choice(['de', 'fr', 'en']),
              help='Langue de recherche (de/fr/en)')
@click.option('--all', is_flag=True, help='Afficher toutes les informations disponibles')
def search(word, lang, all):
    """Recherche un mot dans le dictionnaire"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if all:
        cursor.execute(f"""
            SELECT de, en, fr, wordType, wiktionary_def, dictcc_def, example_de, example_fr
            FROM dictionary 
            WHERE {lang} LIKE ?
            LIMIT 10
        """, (f"%{word}%",))
    else:
        cursor.execute(f"""
            SELECT de, en, fr, wordType
            FROM dictionary 
            WHERE {lang} LIKE ?
            LIMIT 10
        """, (f"%{word}%",))
    
    results = cursor.fetchall()
    if not results:
        click.echo("Mot non trouvé")
        return
    
    for row in results:
        click.echo(format_word_info(row, all))
    
    conn.close()

@cli.command()
@click.option('--all', is_flag=True, help='Afficher toutes les informations disponibles')
def random(all):
    """Affiche un mot aléatoire"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if all:
        cursor.execute("""
            SELECT de, en, fr, wordType, wiktionary_def, dictcc_def, example_de, example_fr
            FROM dictionary 
            ORDER BY RANDOM() 
            LIMIT 1
        """)
    else:
        cursor.execute("""
            SELECT de, en, fr, wordType
            FROM dictionary 
            ORDER BY RANDOM() 
            LIMIT 1
        """)
    
    row = cursor.fetchone()
    if row:
        click.echo(format_word_info(row, all))
    
    conn.close()

@cli.command()
def stats():
    """Affiche les statistiques du dictionnaire"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    click.echo("=" * 50)
    click.echo("STATISTIQUES DU DICTIONNAIRE")
    click.echo("=" * 50)
    
    # Nombre total de mots
    cursor.execute("SELECT COUNT(*) FROM dictionary")
    total = cursor.fetchone()[0]
    click.echo(f"\nNombre total de mots: {total}")
    
    # Distribution par type de mot
    cursor.execute("""
        SELECT wordType, COUNT(*) as count 
        FROM dictionary 
        GROUP BY wordType 
        ORDER BY count DESC
    """)
    click.echo("\nDistribution par type de mot:")
    for type_, count in cursor.fetchall():
        click.echo(f"  {type_}: {count}")
    
    conn.close()

if __name__ == '__main__':
    cli()
