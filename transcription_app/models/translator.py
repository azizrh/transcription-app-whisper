from transformers import MarianMTModel, MarianTokenizer
import torch

class TranslationManager:
    def __init__(self):
        self.language_models = {
            'es': 'Helsinki-NLP/opus-mt-en-es',
            'fr': 'Helsinki-NLP/opus-mt-en-fr',
            'de': 'Helsinki-NLP/opus-mt-en-de',
            'it': 'Helsinki-NLP/opus-mt-en-it'
        }
        self.loaded_models = {}

    def get_model(self, lang_code):
        if lang_code not in self.loaded_models:
            if lang_code not in self.language_models:
                raise ValueError(f"Unsupported language: {lang_code}")
            
            model_name = self.language_models[lang_code]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.to('cuda')
            self.loaded_models[lang_code] = (model, tokenizer)
        
        return self.loaded_models[lang_code]

    def translate(self, text, lang_code):
        if not text.strip():
            return ""
        
        model, tokenizer = self.get_model(lang_code)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]