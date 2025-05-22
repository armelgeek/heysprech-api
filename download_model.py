from transformers import MarianMTModel, MarianTokenizer

model_name = "german-nlp-group/electra-base-german-uncased"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Sauvegarde en local
model.save_pretrained("./electra-german")
tokenizer.save_pretrained("./electra-german")
