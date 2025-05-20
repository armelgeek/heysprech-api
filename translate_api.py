from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer
import shutil
import uuid
import json
import sys
import subprocess
import os

app = FastAPI()

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

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_filename = f"/tmp/{uuid.uuid4()}_{file.filename}"

    with open(tmp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        cmd = [
            sys.executable,
            "-m", "whisper",
            tmp_filename,
            "--model", MODELE,
            "--language", LANGUE,
            "--output_format", FORMAT_SORTIE,
            "--output_dir", "/tmp",
            "--no_speech_threshold", "0.3",
            "--word_timestamps", "True"
        ]
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            return {"error": f"Transcription failed with code {result.returncode}"}

        json_output_path = tmp_filename.rsplit(".", 1)[0] + ".json"
        with open(json_output_path, "r", encoding="utf-8") as f:
            transcription_data = json.load(f)

        return transcription_data

    except Exception as e:
        return {"error": str(e)}

    finally:
        try:
            os.remove(tmp_filename)
            if os.path.exists(json_output_path):
                os.remove(json_output_path)
        except Exception:
            pass