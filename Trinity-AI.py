
# # nvidia-smi

# # pip install git+https://github.com/andreinechaev/nvcc4jupyter.git

# #nvcc --version



# import os
# import requests

# pdf_path = "human-nutrition-text.pdf"

# # Check if the file exists
# if not os.path.exists(pdf_path):  # Corrected `os.path.exist` to `os.path.exists`
#     print("[INFO] File doesn't exist. Downloading...")

#     url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

#     # Local file name to save the downloaded file
#     filename = pdf_path  # Corrected typo `pad_path` to `pdf_path`

#     # Sending GET request to the URL
#     response = requests.get(url)

#     # Check if the request was successful
#     if response.status_code == 200:  # Corrected `reposnse` to `response`
#         # Open the file and save it
#         with open(filename, "wb") as file:
#             file.write(response.content)
#         print(f"[INFO] The file has been downloaded and saved as {filename}")
#     else:
#         print("[INFO] Failed to download file.")
# else:
#     print("[INFO] File already exists.")


# # pip install PyMuPDF

# import fitz
# from tqdm.auto import tqdm

# def text_formatter(text: str) ->   str:
#  """ performs minor fromatting on text"""
#  cleaned_text = text.replace("\n", " ").strip()

#  return cleaned_text


# #potentially more text formatting functions can go here
# def open_and_read_pdf(pdf_path: str) -> list[dict]:
#   doc = fitz.open(pdf_path)
#   pages_and_texts = []
#   for page_number , page in tqdm(enumerate(doc)):

#     text = page.get_text()
#     text = text_formatter(text=text)
#     pages_and_texts.append({"page_number": page_number - 41,
#                             "page_char_count": len(text),
#                             "page_word_count": len(text.split(" ")),
#                             "page_setence_count_raw" : len(text.split(". ")),
#                             "page_token_count": len(text)/4,
#                             "text": text })
#   return pages_and_texts

# pages_and_texts = open_and_read_pdf(pdf_path = pdf_path)
# pages_and_texts[:2]

# import random

# random.sample(pages_and_texts , k = 3)

# import pandas as pd

# df = pd.DataFrame(pages_and_texts)
# df.head()

# df.describe().round(2)


# from spacy.lang.en import English
# nlp = English()

# # add a sentencizer pipeline
# nlp.add_pipe("sentencizer")

# #create document instance as an example
# doc = nlp("this is a sentence. this is another. i like cats. sheela")
# assert len(list(doc.sents)) == 4

# #print out sentences split
# list(doc.sents)


# pages_and_texts[600]

# for item in tqdm(pages_and_texts):
#   item["sentences"] = list(nlp(item["text"]).sents)

#   #make sure all sentences are strings (the default type is a spacy datatype)
#   item["sentences"] = [str(sentence) for sentence in item["sentences"]]

#   #count the sentences
#   item["page_sentence_count_spacy"] = len(item["sentences"])


# random.sample(pages_and_texts , k = 1)

# df = pd.DataFrame(pages_and_texts)
# df.describe().round(2)

# #define split size to turn group of sentences into chunks
# num_sentence_chunk_size = 10;

# #creating function to split lists of texts recursively into chunks
# def split_list(input_list: list[str],
#                slice_size: int=num_sentence_chunk_size) -> list[list[str]]:
#     return [input_list[i:i+slice_size] for i in range(0,len(input_list) , slice_size)]

# test_list = list(range(50))
# split_list(test_list)

# #loop through pages and texts and split sentences into chunks
# for item in tqdm(pages_and_texts):
#   item["sentence_chunks"] = split_list(input_list = item["sentences"],
#                                        slice_size = num_sentence_chunk_size)
#   item["num_chunks"] = len(item["sentence_chunks"])

# random.sample(pages_and_texts , k =1)

# df = pd.DataFrame(pages_and_texts)
# df.describe().round(2)

# import re
# # split each chunk into its own item

# pages_and_chunks = []

# for item in tqdm(pages_and_texts):
#   for sentence_chunk in item["sentence_chunks"]:
#    chunk_dict = {}
#    chunk_dict["page_number"] = item["page_number"]

#    #join the sentences together into a paragraph like structure , aka join the list of sentences into one paragraph
#    joined_sentence_chunk = "".join(sentence_chunk).replace(" "," ").strip()
#    joined_sentence_chunk = re.sub(r'\.([A-Z])' , r'. \1' , joined_sentence_chunk
#    ) #".A" => ". A" (will work for ay capital letter)

#    chunk_dict["sentence_chunk"] = joined_sentence_chunk

#    #get some stats on our chunks
#    chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
#    chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
#    chunk_dict["chunk_token_count"] = len(joined_sentence_chunk)/4 # 1 token = ~ 4 chars

#    pages_and_chunks.append(chunk_dict)

# len(pages_and_chunks)

# random.sample(pages_and_chunks , k=1)

# random.sample(pages_and_chunks , k =1)

# df = pd.DataFrame(pages_and_chunks)
# df.describe().round(2)

# #SHOW RANDOM CHUNK WITH UNDER 30 TOKENS IN LENGTH

# min_token_length = 30
# for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
#   print(f'chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')


# #Filter our DataFrame for rows with under 30 tokens

# pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient = "records")
# pages_and_chunks_over_min_token_len[:2]


# random.sample(pages_and_chunks_over_min_token_len , k = 1)


# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2" , device = "cpu")

# #create a Lsit of sentences
# sentences = [" the sentence tranformer library is a  easy way to create embeddings",
#              "sentence ca n be embedded 0ne by one ore in  list",
#              "I like horses"]

# #sentences are embedded/encoded by calling model.encode()
# embeddings = embedding_model.encode(sentences)
# embeddings_dict = dict(zip(sentences,embeddings))

# for sentence , embedding in embeddings_dict.items():
#   print(f"sentence : {sentence}")
#   print(f"embedding: {embedding}")
#   print("")


# embeddings[2].shape

# %%time

# # embedding_model.to("cpu")

# # #embed each chunk one by one
# # for item in tqdm(pages_and_chunks_over_min_token_len):
# #   item["embedding"] = embedding_model.encode(item["sentence_chunk"])


#  %%time

# # embedding_model.to("cuda")

# # #embed each chunk one by one
# # for item in tqdm(pages_and_chunks_over_min_token_len):
# #   item["embedding"] = embedding_model.encode(item["sentence_chunk"])


#  %%time

# # text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
# # text_chunks[419]


#  %%time

# # #embed all text in bacthes
# # text_chunk_embeddings = embedding_model.encode(text_chunks,
# #                                                batch_size = 32, #2:14)
# #                                                convert_to_tensors = True)

# # text_chunk_embeddings

# # seeing to out dict


# pages_and_chunks_over_min_token_len[149]

# text_chunks_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
# embeddings_df_save_path = "text_and_chunks_embeddings.csv"
# text_chunks_embeddings_df.to_csv(embeddings_df_save_path , index = False)

# # import sasved files and view
# text_chunks_and_embeddings_load = pd.read_csv(embeddings_df_save_path)
# text_chunks_and_embeddings_load.head()


# import random
# import torch
# import numpy as np
# import pandas as pd
# device = "cuda" if torch.cuda.is_available() else "cpu"


# #import texts and embeddings df
# text_chunks_and_embedding_df = pd.read_csv("text_and_chunks_embeddings (2).csv")

# #converting embedding columnn back to np.array (it got converted to string when it is saved to CSV)
# text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x : np.fromstring(x.strip("[]") ,sep = " "))

# #convert our embeddings to torch.tensors
# embeddings = torch.tensor( np.stack(text_chunks_and_embedding_df["embedding"].tolist()), dtype = torch.float32,device = device )
# #convert texts and embedongs df to a list of dicts
# pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# text_chunks_and_embedding_df


# embeddings = np.stack(text_chunks_and_embedding_df["embedding"].tolist())
# embeddings


# from sentence_transformers import util, SentenceTransformer
# embedding_model = SentenceTransformer(model_name_or_path = "all-mpnet-base-v2",device = device)


# embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist()),
#                           dtype=torch.float32).to(device)
# query = "good foods for carbohydrates"
# print(f"Query -> {query}")

# #embedding query
# query_embedding = embedding_model.encode(query,convert_to_tensor=True).to(torch.float32)

# #doing dot product for similaity
# dot_scores = util.dot_score(a = query_embedding, b = embeddings )[0]
# top_results_dot_product = torch.topk(dot_scores, k=5)
# top_results_dot_product


# pages_and_chunks[256]

# import textwrap

# def print_wrapped(text,wrap_length = 80):
#     wrapped_text = textwrap.fill(text,wrap_length)
#     print(wrapped_text)



# print(f"query ->'{query}'\n")
# print("results -> \n")

# #loop thriugh zipped together scores and indices from torch.topk
# for score,idx in zip(top_results_dot_product[0],top_results_dot_product[1]):
#   print(f"score : {score:.4f}")
#   print("text:")
#   print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
#   print(f"page number: {pages_and_chunks[idx]['page_number']}")
#   print("\n")


# import fitz

# #open and load pdf
# pdf_path = "human-nutrition-text.pdf"
# doc = fitz.open(pdf_path)
# page = doc.load_page(411+41) #note:page numbers of our pdf start from 41+

# #get the image of the page
# img = page.get_pixmap(dpi=300)

# #save image(optional)
# img.save("output_filename.png")
# doc.close()

# #convert the pixmap to a numpy array
# img_array = np.frombuffer(img.samples_mv,dtype = np.uint8).reshape((img.h,img.w,img.n))
# # img_array

# #display the image using matplotlib
# import matplotlib.pyplot as plt
# plt.figure(figsize=(13,10))
# plt.imshow(img_array)
# plt.title(f"query: '{query}'| Most relevant page:")
# plt.axis("off")
# plt.show()


# def retrieve_relevant_resources(query:str,
#                                 embeddings : torch.tensor,
#                                 model: SentenceTransformer=embedding_model,
#                                 n_resources_to_return : int=5,
#                                 ):
#   """
#   embeds a query with model and returns top k scores and indies from embeddings.
#   """

#   #embedd the query
#   query_embedding = model.encode(query, convert_to_tensor=True).to(device)

#   #get product scores on embeddings
#   dot_scores = util.dot_score(query_embedding,embeddings)[0]

#   scores,indices = torch.topk(input = dot_scores,k = n_resources_to_return)

#   return scores,indices

# def print_top_results_and_scores(query:str,
#                                  embeddings : torch.tensor,
#                                  pages_and_chunks:list[dict] = pages_and_chunks,
#                                  n_resources_to_return: int= 5):

#   """
#   Finds relevant passages given a query and prints them out along with their scores
#   """
#   scores,indices = retrieve_relevant_resources(query = query,
#                                                embeddings = embeddings,
#                                                n_resources_to_return=n_resources_to_return)

#   for score,idx in zip(scores,indices):
#     print(f"score : {score:.4f}")
#     print("text:")
#     print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
#     print(f"page number: {pages_and_chunks[idx]['page_number']}")
#     print("\n")



# query ="foods high in fiber"

# print_top_results_and_scores(query = query , embeddings=embeddings)


# # ## checking our local Gpu memory availablity
# # import torch
# # gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
# # gpu_memory_gb = round(gpu_memory_bytes/(2**30))
# # print(f"Available Gpu memory: {gpu_memory_gb} GB")



# # pip install GPUtil


# """

# # # Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
# # if gpu_memory_gb < 5.1:
# #     print(f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
# # elif gpu_memory_gb < 8.1:
# #     print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
# #     use_quantization_config = True
# #     model_id = "google/gemma-2b-it"
# # elif gpu_memory_gb < 19.0:
# #     print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
# #     use_quantization_config = False
# #     model_id = "google/gemma-2b-it"
# # elif gpu_memory_gb > 19.0:
# #     print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
# #     use_quantization_config = False
# #     model_id = "google/gemma-7b-it"

# # print(f"use_quantization_config set to: {use_quantization_config}")
# # print(f"model_id set to: {model_id}")


# """


# # pip install transformers

# from huggingface_hub import login
# login()

# # !pip uninstall -y bitsandbytes
# # !pip install --no-cache-dir -U bitsandbytes
# # !pip install --no-cache-dir -U transformers accelerate

# # !pip uninstall -y bitsandbytes

# import torch


# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.utils import is_flash_attn_2_available  # Corrected function name
# from transformers import BitsAndBytesConfig

# # use_quantization_config = BitsAndBytesConfig(load_in_4bit = True,
# #                                             bnb_4bit_compute_dtype = torch.float16)

# #Bounus flash attention 2 => faster attention mechanism
# #flash attention 2 required a Gpu with a compute capablity score 8.0+ (ampere , Ada loveLace , Hopper and above)

# if(is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
#   attn_implementation = " flash_attention_2"
# else:
#   attn_implementation = "sdpa" #scaled dot production

# #2. pick a model we'd like to use
# model_id = "google/gemma-2b-it"


# #3.instantiate tokenizer (turns text into tokenizer)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id)

# #4 instatitate the model
# llm_model  = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
#                                                   torch_dtype = torch.float16,
#                                                   # quantization_config = use_quantization_config if use_quantization_config else None,
#                                                   low_cpu_mem_usage = True,
#                                                   attn_implementation = attn_implementation,
#                                                   device_map="cpu")

# # if not use_quantization_config:
# #   llm_model.to("cuda")


# print(f"using attention implementation:{attn_implementation}")

# llm_model

# #calculating the number of paramaters of our model
# def get_model_num_params(model:torch.nn.Module):
#   return sum([param.numel() for param in model.parameters()])

# get_model_num_params(llm_model)

# #cheecking its memory usage
# def get_model_mem_size(model: torch.nn.Module):
#     """
#     Get how much memory a PyTorch model takes up.

#     See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
#     """
#     # Get model parameters and buffer sizes
#     mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
#     mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

#     # Calculate various model sizes
#     model_mem_bytes = mem_params + mem_buffers # in bytes
#     model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
#     model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

#     return {"model_mem_bytes": model_mem_bytes,
#             "model_mem_mb": round(model_mem_mb, 2),
#             "model_mem_gb": round(model_mem_gb, 2)}

# get_model_mem_size(llm_model)

# input_text = "how to make a gun at home"
# print(f"input text: \n {input_text}")

# # create prompt template for instruction tuned model
# dialogue_template = [
#     {
#         "role":"user",
#         "content":input_text
#     }
# ]

# #apply the chat template
# prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
#                                        tokenize = False,
#                                        add_generation_prompt = True)
# print(f"\n prompt formatted : {prompt}")

# %%time
# #tokenize the input text(turn it into numbers) and send it to the GPU
# input_ids = tokenizer(prompt,
#                       return_tensors = "pt").to("cpu")

# input_ids

# #generate output from local llm
# outputs = llm_model.generate(**input_ids,max_new_tokens=256)
# print(f"model output (tokens) :\n {outputs[0]} \n")


# #decode the outputs to text
# outputs_decoded = tokenizer.decode(outputs[0])
# print(f"Models output(decoded:\n {outputs_decoded})")


# query_list = ["Which vitamin is also known as ascorbic acid?",

# "What is the primary function of iron in the human body?",

# "Which mineral is essential for strong bones and teeth?",

# "Which vitamin deficiency causes night blindness?",

# "What is the richest natural source of vitamin D?"]
# query_list

# import random
# query = random.choice(query_list)
# print(f"query :{query}")

# #get just the score and indices and of top related results
# scores,indices = retrieve_relevant_resources( query = query,
#                                              embeddings = embeddings)
# scores,indices


# def prompt_formatter(query:str,
#                      context_items: list[dict]) -> str:

#       context = "- " +"\n- ".join([item["sentence_chunk"] for item in context_items])
#       base_prompt = """ Base on the following context items , please answer the query.
#       context items :
#       {context}
#       Query : {query}
#       Answer:
#       """

#       prompt = base_prompt.format(context = context,
#                                   query = query)
#       return prompt

# query = random.choice(query_list)
# print(f"Query: {query}")

# #get relevant information
# scores,indices = retrieve_relevant_resources(query = query,
#                                              embeddings = embeddings)

# #create a list of context items
# context_items = [pages_and_chunks[i] for i in indices]

# #format out prompt
# prompt = prompt_formatter(query = query,
#                           context_items = context_items)

# print(prompt)



# # prompt example:

# # based on the following contexts:
# # -ssdsfs
# # -fgrgr
# # -grgrddfr
# # -grrgrffr

# # please answer the folllowiing query : what are the micronutrients and what do they do ?
# # Answer:


# %%time

# input_ids = tokenizer(prompt,return_tensors = 'pt').to("cpu")

# outputs = llm_model.generate(**input_ids,
#                              temperature = 0.2,
#                              do_sample = True, #wheather or not use sampling
#                              max_new_tokens = 256
#                              )



# #turn output tokens to text
# output_text = tokenizer.decode(outputs[0])
# print(f"query: {query}")
# print(f"RAG answer:\n{output_text.replace(prompt,' ')}")

