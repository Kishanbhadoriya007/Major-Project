# summarizer.py (Updated for Flask)
from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)

# --- Cache for loaded pipelines (Simple in-process cache per module) ---
_loaded_summarizer_pipelines = {}

def summarize_text(text: str,
                   max_length: int = 150,
                   min_length: int = 30,
                   chunk_size: int = 1024,
                   device: int = -1, # Pass device setting
                   model_name: str = 'facebook/bart-large-cnn'
                   ) -> str:
    """Summarizes the input text using a cached or newly loaded pipeline."""
    global _loaded_summarizer_pipelines
    pipeline_key = f"summarizer_{model_name}_dev{device}" # Key includes device

    if pipeline_key not in _loaded_summarizer_pipelines:
        logger.info(f"Loading summarization model '{model_name}' on device {device}...")
        try:
            summarizer_pipeline = pipeline("summarization", model=model_name, device=device)
            _loaded_summarizer_pipelines[pipeline_key] = summarizer_pipeline
            logger.info(f"Loaded summarization model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to load summarization model '{model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Summarization model '{model_name}' could not be loaded: {e}") from e
    else:
        summarizer_pipeline = _loaded_summarizer_pipelines[pipeline_key]
        logger.info(f"Using cached summarization model '{model_name}'.")

    if not text:
        logger.info("No text provided for summarization.")
        return "[INFO] No text provided for summarization."

    try:
        logger.info("Starting summarization...")
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_summaries = []
        total_chunks = len(text_chunks)

        for i, chunk in enumerate(text_chunks):
            logger.info(f"Summarizing chunk {i+1}/{total_chunks}...")
            effective_min_length = min(min_length, len(chunk.split()) // 3)
            if effective_min_length < 5: effective_min_length = 5

            # Ensure chunk isn't trivially small compared to max_length asked
            effective_max_length = max(max_length, effective_min_length + 10) # Ensure max > min

            summary = summarizer_pipeline(chunk, max_length=effective_max_length, min_length=effective_min_length, do_sample=False)[0]['summary_text']
            all_summaries.append(summary)

        final_summary = "\n".join(all_summaries)

        if len(final_summary) > max_length * 1.5 and len(text_chunks) > 1: # Only re-summarize if multiple chunks were combined
            logger.info("Summarizing the combined summaries...")
            final_summary = summarizer_pipeline(final_summary, max_length=max_length+50, min_length=min_length, do_sample=False)[0]['summary_text']

        logger.info("Summarization complete.")
        return final_summary

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise RuntimeError(f"Summarization process failed: {e}") from e