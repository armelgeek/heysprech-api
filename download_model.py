from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "dbmdz/german-gpt2"
print(f"Téléchargement du modèle {model_name}...")
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Sauvegarde en local
print("Sauvegarde du modèle en local...")
model.save_pretrained("./german-gpt2")
tokenizer.save_pretrained("./german-gpt2")

print("Modèle téléchargé et sauvegardé avec succès!")
