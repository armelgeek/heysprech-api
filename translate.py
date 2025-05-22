#!/usr/bin/env python3
import json
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm

def load_translation_models():
    """Chargement des modèles de traduction"""
    print("Chargement du modèle de traduction DE-EN...")
    tokenizer_de_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model_de_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    
    print("Chargement du modèle de traduction EN-FR...")
    tokenizer_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    
    return {
        'de_en': (tokenizer_de_en, model_de_en),
        'en_fr': (tokenizer_en_fr, model_en_fr)
    }

def load_text_generator():
    """Chargement du modèle de génération de texte allemand"""
    print("Chargement du modèle de génération de texte allemand...")
    model_name = "benjamin/gpt2-wechsel-german"
    print(f"Chargement du modèle {model_name}...")
    tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model_gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
    tokenizer_gpt2.padding_side = 'left'
    return tokenizer_gpt2, model_gpt2

def translate_text(text, models, source_lang='en'):
    """Traduction d'un texte vers le français
    
    Args:
        text: Texte à traduire
        models: Dictionnaire contenant les modèles de traduction
        source_lang: Langue source ('de' pour allemand, 'en' pour anglais)
    """
    if source_lang == 'de':
        # Traduction allemand -> anglais -> français
        tokenizer_de_en, model_de_en = models['de_en']
        tokenizer_en_fr, model_en_fr = models['en_fr']
        
        # Première traduction : DE -> EN
        inputs_de = tokenizer_de_en(text, return_tensors="pt", padding=True)
        outputs_de = model_de_en.generate(**inputs_de)
        english = tokenizer_de_en.decode(outputs_de[0], skip_special_tokens=True)
        
        # Deuxième traduction : EN -> FR
        inputs_en = tokenizer_en_fr(english, return_tensors="pt", padding=True)
        outputs_en = model_en_fr.generate(**inputs_en)
        return tokenizer_en_fr.decode(outputs_en[0], skip_special_tokens=True)
    else:
        # Traduction directe anglais -> français
        tokenizer_en_fr, model_en_fr = models['en_fr']
        inputs = tokenizer_en_fr(text, return_tensors="pt", padding=True)
        outputs = model_en_fr.generate(**inputs)
        return tokenizer_en_fr.decode(outputs[0], skip_special_tokens=True)

def generate_example_sentences(word, word_type, translation_models, text_generator):
    """Génère et traduit trois phrases d'exemple très simples pour l'apprentissage de la langue"""
    examples = []
    
    try:
        tokenizer_gpt2, model_gpt2 = text_generator
        
        # Prompts selon le type de mot avec des phrases très simples
        if word_type and word_type.lower().startswith('v'):
            prompts = [
                f"Ich {word} gern.",
                f"Er {word} jetzt.",
                f"Wir {word} hier.",
                f"Sie {word} morgen."
            ]
        elif word_type and word_type.lower().startswith('adj'):
            prompts = [
                f"Es ist {word}.",
                f"Sehr {word}!",
                f"Das ist {word}.",
                f"Der Tag ist {word}."
            ]
        else:  # Noms par défaut
            prompts = [
                f"Das {word}!",
                f"Ein {word}, bitte.",
                f"Ich brauche das {word}.",
                f"Wo ist das {word}?"
            ]
    
        # Configuration des modèles de traduction
        tokenizer_de_en, model_de_en = translation_models['de_en']
        tokenizer_en_fr, model_en_fr = translation_models['en_fr']
        
        # Génération et traduction des phrases
        for prompt in prompts:
            # Génération de la phrase en allemand
            inputs = tokenizer_gpt2(prompt, return_tensors="pt", padding=True, truncation=True)
            outputs = model_gpt2.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                do_sample=True
            )
            german_sentence = tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)
            
            # Traduction en anglais
            inputs_de = tokenizer_de_en(german_sentence, return_tensors="pt", padding=True)
            outputs_de = model_de_en.generate(**inputs_de)
            english = tokenizer_de_en.decode(outputs_de[0], skip_special_tokens=True)
            
            # Traduction en français
            inputs_en = tokenizer_en_fr(english, return_tensors="pt", padding=True)
            outputs_en = model_en_fr.generate(**inputs_en)
            french = tokenizer_en_fr.decode(outputs_en[0], skip_special_tokens=True)
            
            examples.append({
                'de': german_sentence,
                'en': english,
                'fr': french
            })
        
        return examples
        
    except Exception as e:
        print(f"Erreur lors de la génération de phrases: {str(e)}")
        return []  # Retourner une liste vide en cas d'erreur

def main():
    # Chargement des modèles
    translation_models = load_translation_models()
    text_generator = load_text_generator()
    
    # Exemple d'utilisation
    words_to_test = [
        ("Haus", "n"),        # nom
        ("schön", "adj"),     # adjectif
        ("gehen", "v"),       # verbe
        ("Auto", "n"),        # nom
        ("schnell", "adj")    # adjectif
    ]
    
    print("Génération d'exemples de phrases...")
    
    for word, word_type in tqdm(words_to_test, desc="Traitement des mots"):
        print(f"\n{'='*50}")
        print(f"Génération d'exemples pour le mot : {word}")
        print(f"Type de mot : {word_type}")
        print(f"{'='*50}")
        
        # Génération des phrases d'exemple
        examples = generate_example_sentences(word, word_type, translation_models, text_generator)
        
        print("Phrases d'exemple générées :")
        for i, example in enumerate(examples, 1):
            print(f"\nExemple {i}:")
            print(f"DE: {example['de']}")
            print(f"EN: {example['en']}")
            print(f"FR: {example['fr']}")
        
        # Sauvegarde au format JSON (optionnel)
        output_file = f"examples_{word}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"\nExemples sauvegardés dans {output_file}")

if __name__ == "__main__":
    main()