import os
from transformers import MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer

# Création des dossiers
print("Création des dossiers de modèles...")
os.makedirs("./opus-mt-de-en", exist_ok=True)
os.makedirs("./opus-mt-en-fr", exist_ok=True)
os.makedirs("./gpt2-german", exist_ok=True)

# Téléchargement et sauvegarde du modèle allemand-anglais
model_name_de_en = "Helsinki-NLP/opus-mt-de-en"
model_de_en = MarianMTModel.from_pretrained(model_name_de_en)
tokenizer_de_en = MarianTokenizer.from_pretrained(model_name_de_en)

print("Sauvegarde du modèle DE-EN...")
model_de_en.save_pretrained("./opus-mt-de-en")
tokenizer_de_en.save_pretrained("./opus-mt-de-en")

# Téléchargement et sauvegarde du modèle anglais-français
model_name_en_fr = "Helsinki-NLP/opus-mt-en-fr"
model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)
tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)

print("Sauvegarde du modèle EN-FR...")
model_en_fr.save_pretrained("./opus-mt-en-fr")
tokenizer_en_fr.save_pretrained("./opus-mt-en-fr")

# Téléchargement du modèle GPT-2 allemand pour la génération de phrases
print("\nTéléchargement du modèle GPT-2 allemand...")
model_name_gpt2_de = "ml6team/gpt2-german"
tokenizer_gpt2_de = AutoTokenizer.from_pretrained(model_name_gpt2_de)
model_gpt2_de = AutoModelForCausalLM.from_pretrained(model_name_gpt2_de)

print("Sauvegarde du modèle GPT-2 allemand...")
model_gpt2_de.save_pretrained("./gpt2-german")
tokenizer_gpt2_de.save_pretrained("./gpt2-german")

print("\nTous les modèles ont été téléchargés et sauvegardés avec succès!")
