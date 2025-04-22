# summarizer.py
from transformers import pipeline
import torch

# Determine device (GPU if available, otherwise CPU)
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"[SUMMARIZER] Using device: {'cuda' if DEVICE == 0 else 'cpu'}")

# Load the summarization pipeline (downloads model on first run)
# Models to try: 'facebook/bart-large-cnn', 'google/pegasus-xsum', 't5-small', 't5-base'
MODEL_NAME = 'facebook/bart-large-cnn'
try:
    summarizer = pipeline("summarization", model=MODEL_NAME, device=DEVICE)
    print(f"[SUMMARIZER] Loaded model '{MODEL_NAME}'.")
except Exception as e:
    print(f"[ERROR] Failed to load summarization model '{MODEL_NAME}': {e}")
    summarizer = None # Indicate failure

def summarize_text(text: str, max_length: int = 150, min_length: int = 30, chunk_size: int = 1024) -> str:
    """Summarizes the input text using a pre-loaded pipeline."""
    if not summarizer:
        return "[ERROR] Summarization model not loaded."
    if not text:
        return "[INFO] No text provided for summarization."

    try:
        print("[INFO] Starting summarization...")
        # Simple chunking based on tokenizer max length (less sophisticated)
        # More robust chunking might split by paragraphs or use sentence boundaries.
        # BART's max input length is often 1024 tokens. Let's use chunk_size for characters as proxy.

        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_summaries = []
        total_chunks = len(text_chunks)

        for i, chunk in enumerate(text_chunks):
            print(f"[INFO] Summarizing chunk {i+1}/{total_chunks}...")
            # Ensure chunk isn't too short for min_length constraint of summary
            effective_min_length = min(min_length, len(chunk.split()) // 3) # Rough heuristic
            if effective_min_length < 5: effective_min_length = 5 # Absolute minimum

            summary = summarizer(chunk, max_length=max_length, min_length=effective_min_length, do_sample=False)[0]['summary_text']
            all_summaries.append(summary)

        final_summary = "\n".join(all_summaries)

        # Optional: If the combined summary is too long, summarize it again
        if len(final_summary) > max_length * 1.5: # If significantly longer than one chunk's max
             print("[INFO] Summarizing the combined summaries...")
             final_summary = summarizer(final_summary, max_length=max_length+50, min_length=min_length, do_sample=False)[0]['summary_text']


        print("[INFO] Summarization complete.")
        return final_summary

    except Exception as e:
        print(f"[ERROR] Summarization failed: {e}")
        # Consider returning partial summary if applicable:
        # if all_summaries: return "\n".join(all_summaries) + "\n[ERROR] Summarization incomplete."
        return "[ERROR] Summarization process failed."