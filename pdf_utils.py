# pdf_utils.py (Updated for Flask)
import fitz  # PyMuPDF
import os
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__) # Use logger for better practice

def extract_text(pdf_path: str) -> str:
    """Extracts all text content from a PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Processing {doc.page_count} pages in {os.path.basename(pdf_path)}...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") + "\n"
        doc.close()
        logger.info("Successfully extracted text.")
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}", exc_info=True)
        # Raise exception for Flask app to catch
        raise ValueError(f"Failed to extract text from PDF: {e}") from e

def extract_images(pdf_path: str, output_dir: str) -> list[str]:
    """
    Extracts images from a PDF file and saves them to the specified output directory.
    Returns a list of absolute paths to the saved images.
    """
    image_paths = []
    if not os.path.exists(pdf_path):
         raise FileNotFoundError(f"Input PDF not found at: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        # Ensure output directory exists (it should be created by the Flask app)
        # os.makedirs(output_dir, exist_ok=True) # App should create it
        if not os.path.isdir(output_dir):
             logger.warning(f"Output directory '{output_dir}' does not exist. Creating it.")
             os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Extracting images from {os.path.basename(pdf_path)} into {output_dir}...")
        img_count = 0
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    # image_ext = base_image["ext"] # We save as PNG

                    # Use Pillow to open and save uniformly as PNG
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        # Handle transparency / convert to RGB
                        if image.mode in ("RGBA", "LA", "P"):
                            bg = Image.new("RGB", image.size, (255, 255, 255))
                            mask = image.getchannel('A') if image.mode[-1] == 'A' else None
                            bg.paste(image, (0, 0), mask)
                            image = bg
                        elif image.mode != "RGB":
                            image = image.convert("RGB")

                        image_filename = f"page{page_num+1}_img{img_index+1}.png"
                        # Save path is absolute within the specific output dir
                        output_path = os.path.join(output_dir, image_filename)
                        image.save(output_path, "PNG")
                        image_paths.append(output_path) # Store absolute path
                        img_count += 1

                    except Exception as pil_e:
                        logger.warning(f"Skipping image on page {page_num+1}, index {img_index+1} due to Pillow processing error: {pil_e}")

                except Exception as extract_e:
                    logger.warning(f"Skipping image on page {page_num+1}, index {img_index+1} due to extraction error: {extract_e}")

        doc.close()
        if img_count > 0:
            logger.info(f"Successfully extracted {img_count} images to '{output_dir}'.")
        else:
            logger.info("No images found or extracted.")
        return image_paths # Return list of absolute paths

    except Exception as e:
        logger.error(f"Failed to process images from {pdf_path}: {e}", exc_info=True)
        # Raise exception
        raise RuntimeError(f"Failed to process images from PDF: {e}") from e