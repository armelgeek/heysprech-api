from transformers import MarianMTModel, MarianTokenizer

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

print("Les modèles ont été téléchargés et sauvegardés avec succès.")
