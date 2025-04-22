# pdf_utils.py
import fitz  # PyMuPDF
import os
from PIL import Image
import io

def extract_text(pdf_path: str) -> str:
    """Extracts all text content from a PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        print(f"[INFO] Processing {doc.page_count} pages in {os.path.basename(pdf_path)}...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") + "\n" # Add newline between pages
        doc.close()
        print(f"[INFO] Successfully extracted text.")
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return ""

def extract_images(pdf_path: str, output_dir: str) -> list[str]:
    """Extracts images from a PDF file and saves them to the output directory."""
    image_paths = []
    try:
        doc = fitz.open(pdf_path)
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
        print(f"[INFO] Extracting images from {os.path.basename(pdf_path)}...")
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
                    image_ext = base_image["ext"]

                    # Convert potential non-standard formats (like jbig2) using Pillow
                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        # Ensure RGB format for broader compatibility if needed, handle transparency
                        if image.mode in ("RGBA", "LA", "P"): # P mode might have transparency
                             # Create a white background image
                            bg = Image.new("RGB", image.size, (255, 255, 255))
                            # Paste the image onto the background using the alpha channel as mask
                            bg.paste(image, (0, 0), image.getchannel('A') if image.mode[-1] == 'A' else None)
                            image = bg
                        elif image.mode != "RGB":
                             image = image.convert("RGB")

                        # Save as PNG for consistency
                        image_filename = f"page{page_num+1}_img{img_index+1}.png"
                        output_path = os.path.join(output_dir, image_filename)
                        image.save(output_path)
                        image_paths.append(output_path)
                        img_count += 1

                    except Exception as pil_e:
                        print(f"[WARN] Skipping image on page {page_num+1}, index {img_index+1} due to Pillow processing error: {pil_e}")
                        # Fallback: save raw bytes if Pillow fails? Less reliable.
                        # image_filename = f"page{page_num+1}_img{img_index+1}_raw.{image_ext}"
                        # output_path = os.path.join(output_dir, image_filename)
                        # with open(output_path, "wb") as f:
                        #     f.write(image_bytes)
                        # image_paths.append(output_path)

                except Exception as extract_e:
                     print(f"[WARN] Skipping image on page {page_num+1}, index {img_index+1} due to extraction error: {extract_e}")


        doc.close()
        if img_count > 0:
             print(f"[INFO] Successfully extracted {img_count} images to '{output_dir}'.")
        else:
             print("[INFO] No images found or extracted.")
        return image_paths

    except Exception as e:
        print(f"[ERROR] Failed to process images from {pdf_path}: {e}")
        return []