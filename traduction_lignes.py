#!/usr/bin/env python3
import os
import sys
import json
from transformers import MarianMTModel, MarianTokenizer

MODEL_PATH = "./opus-mt-de-fr"

EXTENSION_TRANSCRIPTION = ".json"

def trouver_fichiers_transcription(repertoire):
    fichiers = []
    for nom_fichier in os.listdir(repertoire):
        chemin = os.path.join(repertoire, nom_fichier)
        if os.path.isfile(chemin) and nom_fichier.lower().endswith(EXTENSION_TRANSCRIPTION):
            fichiers.append(chemin)
    return fichiers

def charger_modele_traduction():
    print("Chargement du modèle de traduction MarianMT...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
    model = MarianMTModel.from_pretrained(MODEL_PATH)
    return tokenizer, model

def traduire_texte(tokenizer, model, texte):
    inputs = tokenizer(texte, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    traduction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return traduction

def traduire_fichier_json(tokenizer, model, chemin_fichier):
    print(f"Traitement traduction du fichier: {chemin_fichier}")
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Whisper stocke les segments dans data["segments"] (liste de dict avec "text")
    if "segments" not in data:
        print(f"Aucun segment trouvé dans {chemin_fichier}. Rien à traduire.", file=sys.stderr)
        return False

    segments = data["segments"]
    for segment in segments:
        texte_original = segment.get("text", "").strip()
        if texte_original:
            traduction = traduire_texte(tokenizer, model, texte_original)
            segment["translation_fr"] = traduction
        else:
            segment["translation_fr"] = ""

    # Enregistrer un nouveau fichier avec suffixe _traduit.json
    chemin_sortie = chemin_fichier.replace(EXTENSION_TRANSCRIPTION, f"_traduit{EXTENSION_TRANSCRIPTION}")
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fichier traduit sauvegardé sous: {chemin_sortie}")
    return True

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <repertoire_contenant_fichiers_json>", file=sys.stderr)
        sys.exit(1)

    repertoire = sys.argv[1]
    if not os.path.isdir(repertoire):
        print(f"Erreur: '{repertoire}' n'est pas un répertoire valide.", file=sys.stderr)
        sys.exit(1)

    fichiers_json = trouver_fichiers_transcription(repertoire)
    if not fichiers_json:
        print(f"Aucun fichier {EXTENSION_TRANSCRIPTION} trouvé dans '{repertoire}'.", file=sys.stderr)
        sys.exit(1)

    tokenizer, model = charger_modele_traduction()

    succes = 0
    erreurs = 0
    for fichier in fichiers_json:
        if traduire_fichier_json(tokenizer, model, fichier):
            succes += 1
        else:
            erreurs += 1

    print(f"\nTraduction terminée: {succes} réussies, {erreurs} erreurs.")

    if erreurs > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
