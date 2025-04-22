# translator.py
from transformers import pipeline
import torch

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"[TRANSLATOR] Using device: {'cuda' if DEVICE == 0 else 'cpu'}")

# Cache loaded pipelines to avoid reloading for the same language
loaded_pipelines = {}

def translate_text(text: str, target_lang: str, chunk_size: int = 512) -> str:
    """Translates text to the target language using Helsinki-NLP models."""
    if not text:
        return "[INFO] No text provided for translation."

    # Assuming input is English. For other source languages, the model name needs changing.
    # Format: Helsinki-NLP/opus-mt-{src}-{tgt}
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"

    try:
        if model_name not in loaded_pipelines:
            print(f"[INFO] Loading translation model: {model_name}")
            translator = pipeline(f"translation_en_to_{target_lang}", model=model_name, device=DEVICE)
            loaded_pipelines[model_name] = translator
            print(f"[TRANSLATOR] Loaded model '{model_name}'.")
        else:
            translator = loaded_pipelines[model_name]
            print(f"[INFO] Using cached translation model: {model_name}")


        print(f"[INFO] Starting translation to '{target_lang}'...")
        # Simple chunking, models often have ~512 token limit
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_translations = []
        total_chunks = len(text_chunks)

        for i, chunk in enumerate(text_chunks):
             print(f"[INFO] Translating chunk {i+1}/{total_chunks}...")
             # Note: Helsinki models might not have explicit min/max length args like summarizers
             translation = translator(chunk)[0]['translation_text']
             all_translations.append(translation)

        final_translation = "\n".join(all_translations) # Join translated chunks
        print("[INFO] Translation complete.")
        return final_translation

    except Exception as e:
        print(f"[ERROR] Translation failed for model {model_name}: {e}")
        # Handle common errors: model not found
        if "can be loaded" in str(e) or "404" in str(e):
             print(f"[HINT] The language code '{target_lang}' might be unsupported or the model name incorrect.")
             print(f"[HINT] Check available models at https://huggingface.co/Helsinki-NLP")
             return f"[ERROR] No model found for en -> {target_lang}."
        return "[ERROR] Translation process failed."