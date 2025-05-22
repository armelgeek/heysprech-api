from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

# Charger le modèle localement
model_path = "./opus-mt-de-fr"
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
def translate(req: TranslationRequest):
    inputs = tokenizer(req.text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": translated_text}
