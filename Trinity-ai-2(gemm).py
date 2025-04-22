# -*- coding: utf-8 -*-
"""
Python script converted from Jupyter notebook code for PDF processing,
embedding generation (using Sentence Transformers), similarity search,
and RAG implementation with a Gemma LLM.
"""

# --- Prerequisites ---
# Ensure you have the necessary libraries installed:
# pip install requests PyMuPDF tqdm pandas spacy sentence-transformers torch numpy matplotlib huggingface_hub transformers accelerate bitsandbytes GPUtil
# python -m spacy download en_core_web_sm  # Or another small English model for sentencizer

# --- Imports ---
import os
import requests
import fitz  # PyMuPDF
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import textwrap
import matplotlib.pyplot as plt
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import time # Added for potential timing if needed


# --- Configuration ---
PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
PDF_PATH = "human-nutrition-text.pdf"
EMBEDDING_CSV_PATH = "text_and_chunks_embeddings.csv" # Make sure this matches the saved file name
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
LLM_MODEL_ID = "google/gemma-2b-it"
NUM_SENTENCE_CHUNK_SIZE = 10
MIN_TOKEN_LENGTH = 30
N_RESOURCES_TO_RETURN = 5

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Helper Functions ---

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Opens and reads a PDF, extracting text and basic metadata per page."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF '{pdf_path}': {e}")
        return []
    pages_and_texts = []
    print(f"Reading PDF: {pdf_path}")
    for page_number, page in tqdm(enumerate(doc), total=len(doc)):
        try:
            text = page.get_text()
            text = text_formatter(text=text)
            # Note: Adjust page_number offset if needed based on PDF structure
            pages_and_texts.append({
                "page_number": page_number - 41, # Original offset from notebook
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4, # Approx token count
                "text": text
            })
        except Exception as e:
            print(f"Error processing page {page_number}: {e}")
            continue # Skip problematic pages
    print(f"Finished reading PDF. Found {len(pages_and_texts)} pages.")
    return pages_and_texts

def split_list(input_list: list, slice_size: int) -> list[list]:
    """Splits a list into sublists of a specified size."""
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def print_wrapped(text, wrap_length=80):
    """Prints text wrapped to a specified length."""
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int = 5):
    """Embeds a query and returns top k scores and indices from existing embeddings."""
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True).to(embeddings.device) # Ensure device match

    # Get dot product scores
    # Use cosine similarity which is equivalent to dot score for normalized embeddings
    # dot_scores = util.dot_score(query_embedding, embeddings)[0]
    dot_scores = util.cos_sim(query_embedding, embeddings)[0]


    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 embedding_model: SentenceTransformer,
                                 n_resources_to_return: int = 5):
    """Finds relevant passages given a query and prints them out."""
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
        n_resources_to_return=n_resources_to_return
    )

    print(f"\nQuery: '{query}'")
    print("Results:")
    for score, idx in zip(scores, indices):
        print(f"Score: {score:.4f}")
        print("Text:")
        print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[idx]['page_number']}")
        print("-" * 20)

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """Formats a prompt for the LLM using retrieved context."""
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Context items:
{context}

Query: {query}
Answer:"""
    prompt = base_prompt.format(context=context, query=query)
    # Add prompt template specific tokens if needed for the model (e.g., Gemma)
    # This basic template might need adjustment based on LLM requirements.
    return prompt

def get_model_num_params(model: torch.nn.Module):
    """Calculates the total number of parameters in a PyTorch model."""
    return sum(param.numel() for param in model.parameters())

def get_model_mem_size(model: torch.nn.Module):
    """Calculates the memory footprint of a PyTorch model."""
    mem_params = sum(param.nelement() * param.element_size() for param in model.parameters())
    mem_buffers = sum(buf.nelement() * buf.element_size() for buf in model.buffers())
    model_mem_bytes = mem_params + mem_buffers
    model_mem_mb = model_mem_bytes / (1024**2)
    model_mem_gb = model_mem_bytes / (1024**3)
    return {
        "model_mem_bytes": model_mem_bytes,
        "model_mem_mb": round(model_mem_mb, 2),
        "model_mem_gb": round(model_mem_gb, 2)
    }

# --- Main Execution ---

# 1. Download PDF if it doesn't exist
if not os.path.exists(PDF_PATH):
    print(f"[INFO] File '{PDF_PATH}' doesn't exist. Downloading...")
    try:
        response = requests.get(PDF_URL, stream=True) # Use stream=True for large files
        response.raise_for_status() # Raise an exception for bad status codes
        with open(PDF_PATH, "wb") as file:
             for chunk in response.iter_content(chunk_size=8192): # Download in chunks
                file.write(chunk)
        print(f"[INFO] The file has been downloaded and saved as {PDF_PATH}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download file: {e}")
        exit() # Exit if download fails
else:
    print(f"[INFO] File '{PDF_PATH}' already exists.")

# 2. Process PDF: Extract text and structure
pages_and_texts = open_and_read_pdf(pdf_path=PDF_PATH)
if not pages_and_texts:
    print("[ERROR] Could not process PDF. Exiting.")
    exit()

# Display some stats if needed
# df_pages = pd.DataFrame(pages_and_texts)
# print("\n--- PDF Page Stats ---")
# print(df_pages.head())
# print(df_pages.describe().round(2))

# 3. Sentence Segmentation
print("\n[INFO] Splitting text into sentences using spaCy...")
nlp = English()
nlp.add_pipe("sentencizer")

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence).strip() for sentence in item["sentences"] if len(str(sentence).strip()) > 0] # Clean sentences
    item["page_sentence_count_spacy"] = len(item["sentences"])

# Optional: Update DataFrame and show stats
# df_pages = pd.DataFrame(pages_and_texts)
# print("\n--- PDF Page Stats (with spaCy sentences) ---")
# print(df_pages.describe().round(2))
# print(random.sample(pages_and_texts, k=1)) # Show a random sample

# 4. Chunk Sentences
print(f"\n[INFO] Chunking sentences into groups of ~{NUM_SENTENCE_CHUNK_SIZE}...")
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                          slice_size=NUM_SENTENCE_CHUNK_SIZE)
    item["num_chunks"] = len(item["sentence_chunks"])

# Optional: Update DataFrame and show stats
# df_pages = pd.DataFrame(pages_and_texts)
# print("\n--- PDF Page Stats (with chunks) ---")
# print(df_pages.describe().round(2))
# print(random.sample(pages_and_texts, k=1)) # Show a random sample

# 5. Create Chunk-Level Dictionary
print("\n[INFO] Creating chunk-level data...")
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join sentences, handle potential spacing issues after periods
        joined_sentence_chunk = " ".join(sentence_chunk).strip()
        # Basic fix for ".A" -> ". A" (might need refinement)
        joined_sentence_chunk = re.sub(r'\.(?=[A-Z])', '. ', joined_sentence_chunk)

        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Calculate stats for the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # Approx

        pages_and_chunks.append(chunk_dict)

print(f"[INFO] Created {len(pages_and_chunks)} chunks.")

# Optional: Show chunk stats
# df_chunks = pd.DataFrame(pages_and_chunks)
# print("\n--- Chunk Stats ---")
# print(df_chunks.describe().round(2))
# print("\nRandom sample chunk:")
# print(random.sample(pages_and_chunks, k=1))

# 6. Filter Chunks by Token Length
print(f"\n[INFO] Filtering chunks smaller than {MIN_TOKEN_LENGTH} tokens...")
df_chunks = pd.DataFrame(pages_and_chunks) # Create DataFrame for easier filtering
pages_and_chunks_over_min_len = df_chunks[df_chunks["chunk_token_count"] > MIN_TOKEN_LENGTH].to_dict(orient="records")
print(f"[INFO] Number of chunks after filtering: {len(pages_and_chunks_over_min_len)}")

# Optional: Show stats for filtered chunks
# print("\nRandom sample chunk (filtered):")
# print(random.sample(pages_and_chunks_over_min_len, k=1))

# 7. Load or Generate Embeddings
# This script assumes embeddings are pre-generated and saved to CSV.
# If you need to generate them, uncomment the following section.

# --- Embedding Generation (Optional - Run only if CSV doesn't exist) ---
# print(f"\n[INFO] Initializing embedding model: {EMBEDDING_MODEL_NAME} on device: {DEVICE}")
# embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=DEVICE)

# print("[INFO] Generating embeddings for text chunks...")
# text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_len]

# # Use lists for CPU, tensors for GPU might be faster if moving back later
# text_chunk_embeddings = embedding_model.encode(
#     text_chunks,
#     batch_size=32,
#     convert_to_numpy=True, # Use numpy for saving to CSV
#     show_progress_bar=True
# )

# print(f"[INFO] Embedding generation complete. Shape: {text_chunk_embeddings.shape}")

# # Add embeddings to the dictionaries
# for i, item in enumerate(pages_and_chunks_over_min_len):
#      item["embedding"] = text_chunk_embeddings[i]

# # Save embeddings to CSV
# print(f"[INFO] Saving chunks and embeddings to {EMBEDDING_CSV_PATH}...")
# embeddings_df = pd.DataFrame(pages_and_chunks_over_min_len)
# # Convert numpy arrays to string representation for CSV saving if needed,
# # but loading requires special handling (like np.fromstring).
# # It might be better to save in a format like parquet or pickle that handles arrays.
# # For CSV:
# embeddings_df["embedding"] = embeddings_df["embedding"].apply(lambda x: ' '.join(map(str, x))) # Convert array to space-separated string
# embeddings_df.to_csv(EMBEDDING_CSV_PATH, index=False)
# print("[INFO] Saved successfully.")
# exit() # Exit after generating and saving embeddings if this was the goal
# --- End Embedding Generation ---

# --- Load Pre-computed Embeddings ---
print(f"\n[INFO] Loading pre-computed embeddings from: {EMBEDDING_CSV_PATH}")
if not os.path.exists(EMBEDDING_CSV_PATH):
    print(f"[ERROR] Embedding file not found: {EMBEDDING_CSV_PATH}")
    print("Please generate embeddings first (uncomment the 'Embedding Generation' section) or provide the correct path.")
    exit()

try:
    text_chunks_and_embedding_df = pd.read_csv(EMBEDDING_CSV_PATH)

    # Convert embedding string back to numpy array, then to torch tensor
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ") # Adjust sep if needed (e.g., ',' if comma-separated)
    )

    # Prepare data structures
    embeddings = torch.tensor(
        np.stack(text_chunks_and_embedding_df["embedding"].tolist()),
        dtype=torch.float32
    ).to(DEVICE) # Move embeddings to the target device

    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records") # Keep original structure name

    print(f"[INFO] Loaded {len(pages_and_chunks)} chunks and embeddings.")
    print(f"[INFO] Embeddings tensor shape: {embeddings.shape}")

except Exception as e:
    print(f"[ERROR] Failed to load or process embeddings from {EMBEDDING_CSV_PATH}: {e}")
    exit()

# 8. Initialize Embedding Model (for querying)
print(f"\n[INFO] Initializing embedding model for querying: {EMBEDDING_MODEL_NAME} on device: {DEVICE}")
embedding_model = SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME, device=DEVICE)

# 9. Perform Similarity Search Example
query = "good foods for carbohydrates"
print_top_results_and_scores(
    query=query,
    embeddings=embeddings,
    pages_and_chunks=pages_and_chunks,
    embedding_model=embedding_model,
    n_resources_to_return=N_RESOURCES_TO_RETURN
)

# 10. Visualize Relevant Page Example
print("\n[INFO] Visualizing the most relevant page for the query...")
# Get the top result's index
scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, model=embedding_model, n_resources_to_return=1)
if len(indices) > 0:
    top_idx = indices[0].item() # Get index as integer
    page_num_to_show = pages_and_chunks[top_idx]['page_number']
    # Adjust page number based on the offset used during processing and fitz's 0-based index
    fitz_page_index = page_num_to_show + 41 # Reverse the offset

    try:
        doc = fitz.open(PDF_PATH)
        if 0 <= fitz_page_index < len(doc):
            page = doc.load_page(fitz_page_index)
            pix = page.get_pixmap(dpi=200) # Lower DPI for faster processing/smaller image
            doc.close()

            # Convert pixmap to numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.h, pix.w, pix.n))

            # Display the image
            plt.figure(figsize=(10, 14))
            plt.imshow(img_array)
            plt.title(f"Query: '{query}' | Most relevant page: {page_num_to_show} (Fitz index: {fitz_page_index})")
            plt.axis("off")
            print("[INFO] Displaying plot window (close it to continue)...")
            plt.show()
        else:
            print(f"[WARN] Calculated page index {fitz_page_index} is out of bounds for the PDF (0-{len(doc)-1}).")
            if doc: doc.close()
    except Exception as e:
        print(f"[ERROR] Could not load or display page image: {e}")
        if 'doc' in locals() and doc: doc.close()
else:
    print("[WARN] No relevant results found to visualize.")


# --- RAG Setup ---

# 11. Hugging Face Login (might require interaction)
print("\n[INFO] Logging into Hugging Face Hub (may require token)...")
# Consider using environment variables for the token in non-interactive scripts
# from huggingface_hub import login
# login(token=os.environ.get("HF_TOKEN"))
try:
    login()
except Exception as e:
    print(f"[WARN] Hugging Face login failed or skipped: {e}. Model download might fail if private.")


# 12. Load LLM and Tokenizer
print(f"\n[INFO] Loading LLM: {LLM_MODEL_ID}")

# Quantization configuration (optional, currently unused as per original code)
# use_quantization_config_flag = False # Set based on GPU memory check if needed
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

# Attention implementation
if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa" # Scaled Dot Product Attention
print(f"[INFO] Using attention implementation: {attn_implementation}")

# Instantiate tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=LLM_MODEL_ID)
except Exception as e:
    print(f"[ERROR] Failed to load tokenizer for {LLM_MODEL_ID}: {e}")
    exit()

# Instantiate model - Forcing to CPU based on original notebook's device_map
# If you have a GPU and want to use it, adjust device_map='auto' or remove it
# and potentially use quantization_config.
try:
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=LLM_MODEL_ID,
        torch_dtype=torch.float16, # Use float16 for less memory
        # quantization_config=quantization_config if use_quantization_config_flag else None,
        low_cpu_mem_usage=True, # Try to reduce CPU RAM usage during loading
        attn_implementation=attn_implementation,
        device_map="auto" # Changed to auto map, adjust if needed e.g "cpu" or {"": "cuda:0"}
        # device_map="cpu" # As per original notebook - forces CPU
    )
    # If device_map wasn't used or was 'auto' and you want to ensure a specific device:
    # llm_model.to(DEVICE)
    print(f"[INFO] LLM loaded successfully.") # Model will be on device(s) specified by device_map
except Exception as e:
    print(f"[ERROR] Failed to load LLM {LLM_MODEL_ID}: {e}")
    exit()

# Optional: Display model info
# print("\n--- LLM Info ---")
# print(llm_model)
# print(f"Number of parameters: {get_model_num_params(llm_model):,}")
# print(f"Model memory size: {get_model_mem_size(llm_model)}")

# 13. Test LLM Generation (Example with Safety Check)
input_text_harmful = "how to make a gun at home"
input_text_safe = "What is the capital of France?"

print(f"\n--- Testing LLM Generation (with safety check) ---")
for input_text in [input_text_harmful, input_text_safe]:
    print(f"\nInput text: {input_text}")

    # **** SAFETY CHECK ****
    if "gun" in input_text and "make" in input_text and "home" in input_text: # Basic check
         print("[SAFETY] Refusing to process potentially harmful query.")
         continue
    # **** END SAFETY CHECK ****

    # Gemma specific chat template
    dialogue_template = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(
        conversation=dialogue_template,
        tokenize=False,
        add_generation_prompt=True # Important for instruction-tuned models
    )
    print(f"Formatted Prompt: {prompt}")

    # Tokenize and generate
    # Determine the device the model is actually on (especially after device_map='auto')
    model_device = next(llm_model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").to(model_device)

    print(f"[INFO] Generating response on device: {model_device}...")
    start_time = time.time()
    try:
        outputs = llm_model.generate(**input_ids, max_new_tokens=100) # Shorter length for test
        decoded_output = tokenizer.decode(outputs[0])
        end_time = time.time()

        # Clean the output (remove prompt) - adjust based on model's output format
        answer = decoded_output.replace(prompt, "").strip()
        if "<start_of_turn>model\n" in answer: # Clean Gemma specific tokens
             answer = answer.split("<start_of_turn>model\n", 1)[1]
        if answer.startswith("Answer:"):
            answer = answer.split("Answer:", 1)[1]


        print(f"Model Output (decoded):\n{answer.strip()}")
        print(f"Generation time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        # If CUDA OOM, suggest reducing batch size, using quantization, or a smaller model.
        if "CUDA out of memory" in str(e):
            print("[Hint] CUDA out of memory. Try reducing max_new_tokens, enabling quantization, or using a smaller model/CPU.")


# 14. RAG Implementation Loop
print("\n--- Starting RAG ---")
query_list = [
    "Which vitamin is also known as ascorbic acid?",
    "What is the primary function of iron in the human body?",
    "Which mineral is essential for strong bones and teeth?",
    "Which vitamin deficiency causes night blindness?",
    "What is the richest natural source of vitamin D?",
    "foods high in fiber", # Added from example
    "good foods for protein" # New example
]

for query in query_list:
    print(f"\n--- Processing Query: '{query}' ---")

    # a. Retrieve relevant context
    print("[RAG] Retrieving relevant context...")
    scores, indices = retrieve_relevant_resources(
        query=query,
        embeddings=embeddings,
        model=embedding_model,
        n_resources_to_return=N_RESOURCES_TO_RETURN
    )
    context_items = [pages_and_chunks[i.item()] for i in indices] # Get items using indices

    # Print retrieved context for verification (optional)
    # print("[RAG] Retrieved Context:")
    # for i, item in enumerate(context_items):
    #     print(f"  {i+1}. Score: {scores[i]:.4f} (Page: {item['page_number']})")
    #     print_wrapped(f"     {item['sentence_chunk']}", wrap_length=75)

    # b. Format prompt with context
    print("[RAG] Formatting prompt...")
    # Use the specific chat template for Gemma
    rag_prompt_content = prompt_formatter(query=query, context_items=context_items)
    dialogue_template_rag = [{"role": "user", "content": rag_prompt_content}]
    final_prompt = tokenizer.apply_chat_template(
         conversation=dialogue_template_rag,
         tokenize=False,
         add_generation_prompt=True
    )
    # print(f"[RAG] Final Prompt for LLM:\n{final_prompt}") # Optional: print the full prompt

    # c. Generate answer with LLM
    print("[RAG] Generating answer using LLM...")
    model_device = next(llm_model.parameters()).device # Re-check device just in case
    input_ids = tokenizer(final_prompt, return_tensors="pt").to(model_device)

    start_time = time.time()
    try:
        outputs = llm_model.generate(
            **input_ids,
            max_new_tokens=256, # Max length of the generated answer
            temperature=0.7,    # Control randomness (lower = more deterministic)
            do_sample=True,     # Enable sampling
            top_p=0.95,         # Nucleus sampling probability
            top_k=50            # Consider top k tokens
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()

        # Clean the output string to get only the answer
        # This depends heavily on the model and the chat template used
        # Find the start of the model's response in the template output
        answer_start_token = "<start_of_turn>model\n" # Common for Gemma
        if answer_start_token in output_text:
            answer = output_text.split(answer_start_token, 1)[1]
        else:
             # Fallback: try to remove the prompt part (might be less reliable)
             # This requires knowing the exact structure after apply_chat_template
             # For simplicity, let's just show the raw useful part if the marker isn't found
             # This part might need adjustment based on observing the actual `output_text`
             answer = output_text # Show everything if marker not found

        # Further cleaning if the model repeats parts of the prompt
        base_prompt_part_to_remove = "Based on the following context items, please answer the query."
        if base_prompt_part_to_remove in answer:
            answer = answer.split(base_prompt_part_to_remove)[-1] # Take the part after the prompt text

        # Remove the explicit "Answer:" if the prompt formatter includes it and the model repeats it
        if answer.strip().startswith("Answer:"):
             answer = answer.split("Answer:",1)[1]


        print(f"\nQuery: {query}")
        print(f"RAG Answer (Generated in {end_time - start_time:.2f}s):\n")
        print_wrapped(answer.strip())
        print("-" * 30)

    except Exception as e:
        print(f"[ERROR] RAG generation failed for query '{query}': {e}")
        if "CUDA out of memory" in str(e):
            print("[Hint] CUDA out of memory. Try reducing max_new_tokens, enabling quantization, or using a smaller model/CPU.")
        continue # Move to the next query

print("\n--- Script Finished ---")