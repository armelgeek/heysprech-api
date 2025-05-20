from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained("./opus-mt-en-fr")
tokenizer = MarianTokenizer.from_pretrained("./opus-mt-en-fr")

# Exemple de traduction
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)

print(output) 
