# image_identifier.py (Updated for Flask)
from transformers import pipeline
from PIL import Image
import torch
import os
import logging

logger = logging.getLogger(__name__)

# --- Cache for loaded pipelines (Simple in-process cache per module) ---
_loaded_captioner_pipelines = {}

def identify_image_batch(image_paths: list[str],
                         device: int = -1, # Pass device setting
                         model_name: str = 'Salesforce/blip-image-captioning-base'
                         ) -> dict[str, str]:
    """Generates captions for a list of image files (absolute paths)."""
    global _loaded_captioner_pipelines
    pipeline_key = f"captioner_{model_name}_dev{device}" # Key includes device

    if pipeline_key not in _loaded_captioner_pipelines:
        logger.info(f"Loading image captioning model '{model_name}' on device {device}...")
        try:
            # Check if Pillow is available first
            _ = Image.new('RGB', (1, 1))

            captioner = pipeline("image-to-text", model=model_name, device=device)
            _loaded_captioner_pipelines[pipeline_key] = captioner
            logger.info(f"Loaded image model '{model_name}'.")
        except ImportError:
             logger.error("Pillow library not found or corrupted. Image identification requires Pillow.", exc_info=True)
             raise RuntimeError("Pillow library not found or corrupted. Cannot perform image identification.")
        except Exception as e:
            logger.error(f"Failed to load image captioning model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Image captioning model '{model_name}' could not be loaded: {e}") from e
    else:
        captioner = _loaded_captioner_pipelines[pipeline_key]
        logger.info(f"Using cached image model '{model_name}'.")


    if not image_paths:
        logger.info("No image paths provided for identification.")
        return {"info": "No image paths provided."}

    results = {}
    logger.info(f"Identifying {len(image_paths)} images...")
    total_images = len(image_paths)

    for i, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path) # Use filename as key
        logger.info(f"Processing image {i+1}/{total_images}: {base_name}...")
        try:
            if not os.path.exists(img_path):
                 raise FileNotFoundError(f"Image file not found: {img_path}")

            img = Image.open(img_path).convert("RGB") # Convert to RGB
            caption = captioner(img)[0]['generated_text']
            results[base_name] = caption
            logger.info(f"  -> Caption: {caption}")

        except FileNotFoundError as fnf_e:
             logger.warning(f"Image file not found: {img_path}")
             results[base_name] = f"[ERROR] File not found: {fnf_e}"
        except Exception as e:
            logger.error(f"Failed to identify image {base_name}: {e}", exc_info=True)
            results[base_name] = f"[ERROR] Processing failed: {e}"

    logger.info("Image identification complete.")
    return results