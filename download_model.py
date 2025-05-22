from transformers import MarianMTModel, MarianTokenizer

model_name = "ml6team/gpt2-german"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Sauvegarde en local
model.save_pretrained("./gpt2-german")
tokenizer.save_pretrained("./gpt2-german")
