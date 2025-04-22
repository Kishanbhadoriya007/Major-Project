# main.py
import argparse
import os
import json
from datetime import datetime

# Import functions from other modules
from pdf_utils import extract_text, extract_images
from summarizer import summarize_text
from translator import translate_text
from image_identifier import identify_image_batch

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Process PDF files for summarization, translation, or image identification.")

parser.add_argument("pdf_path", help="Path to the input PDF file.")
parser.add_argument("--mode", choices=['summarize', 'translate', 'identify_images'], required=True,
                    help="Operation mode: summarize text, translate text, or identify images.")
parser.add_argument("--output_dir", default="output",
                    help="Directory to save results (default: ./output).")

# Translation specific arguments
parser.add_argument("--target_lang", default="fr",
                    help="Target language code for translation (e.g., 'fr', 'es', 'de', 'ja'). Required for --mode translate.")

# Summarization specific arguments
parser.add_argument("--max_summary_length", type=int, default=250,
                    help="Maximum length of the summary (default: 250 tokens).")
parser.add_argument("--min_summary_length", type=int, default=50,
                    help="Minimum length of the summary (default: 50 tokens).")

# Chunk size for processing large texts (summarization/translation)
parser.add_argument("--chunk_size", type=int, default=1024,
                   help="Approximate character chunk size for processing large texts (default: 1024 for summarizer, 512 for translator).")


args = parser.parse_args()

# --- Main Logic ---
def main():
    start_time = datetime.now()
    print(f"\n--- Starting PDF Processor ---")
    print(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Input PDF: {args.pdf_path}")
    print(f"Output Directory: {args.output_dir}")

    if not os.path.exists(args.pdf_path):
        print(f"[ERROR] Input PDF file not found: {args.pdf_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.pdf_path))[0]

    # --- Mode Execution ---
    if args.mode == 'summarize':
        print("\n--- Summarization Mode ---")
        text = extract_text(args.pdf_path)
        if text:
            summary = summarize_text(text,
                                     max_length=args.max_summary_length,
                                     min_length=args.min_summary_length,
                                     chunk_size=args.chunk_size) # Use args.chunk_size
            output_path = os.path.join(args.output_dir, f"{base_filename}_summary.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"--- Summary of {os.path.basename(args.pdf_path)} ---\n\n")
                f.write(summary)
            print(f"[SUCCESS] Summary saved to: {output_path}")
        else:
            print("[ERROR] Could not extract text for summarization.")

    elif args.mode == 'translate':
        print("\n--- Translation Mode ---")
        print(f"Target Language: {args.target_lang}")
        text = extract_text(args.pdf_path)
        if text:
             # Translator uses smaller default chunk size, let's adjust if user didn't override
             translator_chunk_size = args.chunk_size if args.chunk_size != 1024 else 512
             translation = translate_text(text,
                                        target_lang=args.target_lang,
                                        chunk_size=translator_chunk_size)
             output_path = os.path.join(args.output_dir, f"{base_filename}_translation_{args.target_lang}.txt")
             with open(output_path, "w", encoding="utf-8") as f:
                 f.write(f"--- Translation ({args.target_lang}) of {os.path.basename(args.pdf_path)} ---\n\n")
                 f.write(translation)
             print(f"[SUCCESS] Translation saved to: {output_path}")
        else:
            print("[ERROR] Could not extract text for translation.")


    elif args.mode == 'identify_images':
        print("\n--- Image Identification Mode ---")
        image_output_dir = os.path.join(args.output_dir, f"{base_filename}_images")
        image_paths = extract_images(args.pdf_path, image_output_dir)

        if image_paths:
            identifications = identify_image_batch(image_paths)
            output_path = os.path.join(args.output_dir, f"{base_filename}_image_captions.json")
            with open(output_path, "w", encoding="utf-8") as f:
                 json.dump(identifications, f, indent=4)
            print(f"[SUCCESS] Image captions saved to: {output_path}")
            print(f"Extracted images are in: {image_output_dir}")
        else:
             print("[INFO] No images were extracted or identified.")


    end_time = datetime.now()
    print(f"\n--- Processor Finished ---")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {end_time - start_time}")

if __name__ == "__main__":
    main()