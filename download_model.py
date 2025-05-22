import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME

# Définir le dossier cache
cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)

model_name = "benjamin/gpt2-wechsel-german"  # Alternative model plus stable
print(f"Téléchargement du modèle {model_name}...")

# Téléchargement avec cache explicite
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
tokenizer.pad_token = tokenizer.eos_token

# Sauvegarde en local
print("Sauvegarde du modèle en local...")
model.save_pretrained("./german-gpt2")
tokenizer.save_pretrained("./german-gpt2")

print("Modèle téléchargé et sauvegardé avec succès!")
