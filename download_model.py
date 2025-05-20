from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-de-fr"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Sauvegarde en local
model.save_pretrained("./opus-mt-de-fr")
tokenizer.save_pretrained("./opus-mt-de-fr")
