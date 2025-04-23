# app.py
import os
import json
import torch
import logging
from datetime import datetime
from flask import (Flask, render_template, request, redirect, url_for,
                   send_from_directory, flash) # flash is optional but good practice
from werkzeug.utils import secure_filename

# Import your refactored functions
from pdf_utils import extract_text, extract_images
from summariser import summarize_text
from translator import translate_text
from image_identifier import identify_image_batch

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Save uploaded PDFs here
OUTPUT_FOLDER = os.path.join('static', 'output') # Save results accessible via URL
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit for uploads
app.config['SECRET_KEY'] = 'your-super-secret-key-for-demo' # Change this ideally

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Ensure Folders Exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- Global AI Device Setting ---
DEVICE = 0 if torch.cuda.is_available() else -1
logger.info(f"Using device: {'cuda' if DEVICE == 0 else 'cpu'}")

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_and_process():
    if request.method == 'POST':
        # --- File Handling ---
        if 'pdf_file' not in request.files:
            flash('No file part') # Optional: flash message
            logger.warning("No file part in request.")
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file') # Optional: flash message
            logger.warning("No file selected.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            start_time = datetime.now()
            # Secure filename and create unique names/dirs
            original_filename = secure_filename(file.filename)
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            unique_id = f"{os.path.splitext(original_filename)[0]}_{timestamp}"

            # Save uploaded PDF temporarily
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}.pdf")
            file.save(upload_path)
            logger.info(f"PDF uploaded and saved to: {upload_path}")

            # Create unique output directory for this request
            request_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], unique_id)
            os.makedirs(request_output_dir, exist_ok=True)
            logger.info(f"Created output directory: {request_output_dir}")

            # --- Get Form Data ---
            mode = request.form.get('mode')
            target_lang = request.form.get('target_lang', 'fr') # Default fr
            try:
                max_summary_len = int(request.form.get('max_summary_length', 250))
                min_summary_len = int(request.form.get('min_summary_length', 50))
            except ValueError:
                 flash('Invalid number format for summary lengths.')
                 return redirect(request.url)

            # --- Processing ---
            results = None
            error_message = None
            output_files = {} # Dict: {type: relative_url}
            image_urls = [] # List of relative URLs for display

            try:
                if mode == 'summarize':
                    logger.info(f"Processing mode: summarize (min:{min_summary_len}, max:{max_summary_len})")
                    text = extract_text(upload_path)
                    if text:
                        summary = summarize_text(
                            text,
                            max_length=max_summary_len,
                            min_length=min_summary_len,
                            device=DEVICE
                        )
                        results = summary
                        # Save result to file
                        output_filename = "summary.txt"
                        output_filepath = os.path.join(request_output_dir, output_filename)
                        with open(output_filepath, "w", encoding="utf-8") as f:
                            f.write(summary)
                        # Get URL relative to static folder
                        output_files['summary'] = url_for('static', filename=f'output/{unique_id}/{output_filename}')
                    else:
                         results = "Could not extract text from PDF."

                elif mode == 'translate':
                    logger.info(f"Processing mode: translate (lang:{target_lang})")
                    text = extract_text(upload_path)
                    if text:
                        translation = translate_text(
                            text,
                            target_lang=target_lang,
                            device=DEVICE
                        )
                        results = translation
                        # Save result to file
                        output_filename = f"translation_{target_lang}.txt"
                        output_filepath = os.path.join(request_output_dir, output_filename)
                        with open(output_filepath, "w", encoding="utf-8") as f:
                             f.write(translation)
                        output_files['translation'] = url_for('static', filename=f'output/{unique_id}/{output_filename}')

                    else:
                         results = "Could not extract text from PDF."

                elif mode == 'identify_images':
                    logger.info("Processing mode: identify_images")
                    # Image output subdir within the request's output dir
                    image_extract_dir = os.path.join(request_output_dir, "extracted_images")
                    os.makedirs(image_extract_dir, exist_ok=True) # Make sure sub-dir exists

                    # extract_images now saves images to image_extract_dir and returns absolute paths
                    extracted_image_paths_abs = extract_images(upload_path, image_extract_dir)

                    if extracted_image_paths_abs:
                        identifications = identify_image_batch(extracted_image_paths_abs, device=DEVICE)
                        results = json.dumps(identifications, indent=4) # Store as JSON string

                        # Save captions to file
                        output_filename = "image_captions.json"
                        output_filepath = os.path.join(request_output_dir, output_filename)
                        with open(output_filepath, "w", encoding="utf-8") as f:
                             json.dump(identifications, f, indent=4)
                        output_files['captions'] = url_for('static', filename=f'output/{unique_id}/{output_filename}')

                        # Generate relative URLs for extracted images for template display
                        for img_abs_path in extracted_image_paths_abs:
                            img_filename = os.path.basename(img_abs_path)
                            # Path relative to static output dir
                            relative_image_url = url_for('static', filename=f'output/{unique_id}/extracted_images/{img_filename}')
                            image_urls.append(relative_image_url)
                    else:
                        results = "No images found or extracted from the PDF."

                else:
                    error_message = "Invalid processing mode selected."

            except Exception as e:
                logger.error(f"Error during processing (mode: {mode}): {e}", exc_info=True)
                error_message = f"An error occurred: {e}"
                results = None # Clear results on error

            finally:
                 # Clean up uploaded file (optional - keep for debugging during presentation?)
                 # try:
                 #      if os.path.exists(upload_path):
                 #           os.remove(upload_path)
                 #           logger.info(f"Removed uploaded file: {upload_path}")
                 # except OSError as e:
                 #      logger.error(f"Error removing uploaded file {upload_path}: {e}")
                 pass

            duration = datetime.now() - start_time
            logger.info(f"Processing finished in {duration}. Results: {results is not None}, Error: {error_message}")

            # Render the template again, now with results
            return render_template('upload.html',
                                   results=results,
                                   error_message=error_message,
                                   duration=str(duration).split('.')[0], # Nicer duration format
                                   output_files=output_files,
                                   image_urls=image_urls,
                                   processed_filename=original_filename,
                                   selected_mode=mode) # Pass mode back

        else:
            flash('Invalid file type. Please upload a PDF.') # Optional
            logger.warning("Invalid file type uploaded.")
            return redirect(request.url)

    # --- Initial GET request ---
    return render_template('upload.html') # Show the initial form


# --- Route to serve output files (needed for downloads if not in static) ---
# If files are in static, Flask serves them automatically via url_for('static', ...)
# If you save output elsewhere, you'd need a route like this:
# @app.route('/download/<path:filename>')
# def download_file(filename):
#     # Construct the full path securely!
#     # Be very careful with directory traversal here in production
#     file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
#     if os.path.exists(file_path):
#          return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
#     else:
#          return "File not found", 404


# --- Run the App ---
if __name__ == '__main__':
    # Consider host='0.0.0.0' if you need access from other devices on your network
    app.run(debug=True, port=5001) # Use a different port if 5000 is busy