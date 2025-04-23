# translator.py (Updated for Flask)
from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)

# --- Cache for loaded pipelines (Simple in-process cache per module) ---
_loaded_translator_pipelines = {}

def translate_text(text: str,
                   target_lang: str,
                   chunk_size: int = 512, # Default chunk size for translation
                   device: int = -1 # Pass device setting
                   ) -> str:
    """Translates text to the target language using Helsinki-NLP models."""
    global _loaded_translator_pipelines

    if not text:
        logger.info("No text provided for translation.")
        return "[INFO] No text provided for translation."

    # Assuming input is English. For other source languages, logic needs changing.
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    pipeline_key = f"translator_{model_name}_dev{device}" # Key includes device
    task = f"translation_en_to_{target_lang}"

    if pipeline_key not in _loaded_translator_pipelines:
        logger.info(f"Loading translation model: {model_name} on device {device}")
        try:
            translator = pipeline(task, model=model_name, device=device)
            _loaded_translator_pipelines[pipeline_key] = translator
            logger.info(f"Loaded translation model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load translation model {model_name}: {e}", exc_info=True)
            # Check for specific Hugging Face model not found errors
            if "can be loaded" in str(e) or "404" in str(e) or "onnx" in str(e).lower():
                 hint = (f"The language code '{target_lang}' might be unsupported or the model name incorrect. "
                         f"Check available models at https://huggingface.co/Helsinki-NLP")
                 logger.error(hint)
                 raise ValueError(f"No translation model found for en -> {target_lang}. {hint}") from e
            raise RuntimeError(f"Translation model '{model_name}' could not be loaded: {e}") from e
    else:
        translator = _loaded_translator_pipelines[pipeline_key]
        logger.info(f"Using cached translation model: {model_name}")

    try:
        logger.info(f"Starting translation to '{target_lang}'...")
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_translations = []
        total_chunks = len(text_chunks)

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Translating chunk {i+1}/{total_chunks}...")
            # Handle potential empty chunks just in case
            if not chunk.strip():
                continue
            translation = translator(chunk.strip())[0]['translation_text']
            all_translations.append(translation)

        final_translation = "\n".join(all_translations)
        logger.info("Translation complete.")
        return final_translation

    except Exception as e:
        logger.error(f"Translation failed for model {model_name}: {e}", exc_info=True)
        raise RuntimeError(f"Translation process failed for {model_name}: {e}") from e