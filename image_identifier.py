# image_identifier.py
from transformers import pipeline
from PIL import Image
import torch
import os

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"[IMAGE_IDENTIFIER] Using device: {'cuda' if DEVICE == 0 else 'cpu'}")

# Using an image captioning model
# Models to try: 'Salesforce/blip-image-captioning-large', 'Salesforce/blip-image-captioning-base', 'nlpconnect/vit-gpt2-image-captioning'
MODEL_NAME = 'Salesforce/blip-image-captioning-base' # Base model is smaller/faster
try:
    # Check if Pillow is available, required by image pipelines
    _ = Image.new('RGB', (60, 30), color = 'red')

    captioner = pipeline("image-to-text", model=MODEL_NAME, device=DEVICE)
    print(f"[IMAGE_IDENTIFIER] Loaded model '{MODEL_NAME}'.")
except ImportError:
    print("[ERROR] Pillow library not found or corrupted. Image identification requires Pillow.")
    captioner = None
except Exception as e:
    print(f"[ERROR] Failed to load image captioning model '{MODEL_NAME}': {e}")
    captioner = None

def identify_image_batch(image_paths: list[str]) -> dict[str, str]:
    """Generates captions for a list of image files."""
    if not captioner:
        return {"error": "Image captioning model not loaded."}
    if not image_paths:
        return {"info": "No image paths provided."}

    results = {}
    print(f"[INFO] Identifying {len(image_paths)} images...")
    total_images = len(image_paths)

    for i, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        print(f"[INFO] Processing image {i+1}/{total_images}: {base_name}...")
        try:
            # Open image using Pillow (ensure compatibility)
            img = Image.open(img_path).convert("RGB") # Convert to RGB
            caption = captioner(img)[0]['generated_text']
            results[base_name] = caption
            print(f"  -> Caption: {caption}")
        except FileNotFoundError:
            print(f"[WARN] Image file not found: {img_path}")
            results[base_name] = "[ERROR] File not found"
        except Exception as e:
            print(f"[ERROR] Failed to identify image {base_name}: {e}")
            results[base_name] = f"[ERROR] Processing failed: {e}"

    print("[INFO] Image identification complete.")
    return results