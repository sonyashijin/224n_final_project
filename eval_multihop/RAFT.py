import pickle
from langchain_community.document_loaders import WikipediaLoader
from transformers import pipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import csv
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
import logging
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
logging.info("This is an info message")

checkpoint_path = "~/final_project/peft-training/checkpoint-1250"
# Load documents from pickle file
with open('loaded_articles_2024_new.pkl', 'rb') as f:
    documents = pickle.load(f)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
device_map = {"": 0}
base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                  device_map='auto',
                                                  quantization_config=bnb_config,
                                                  trust_remote_code=True,
                                                  use_auth_token=hf_token)

ft_model = PeftModel.from_pretrained(base_model, checkpoint_path, torch_dtype=torch.float16, is_trainable=False)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen(model, p, maxlen=100, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample, num_return_sequences=1, temperature=0.1, num_beams=1, top_p=0.95).to('cpu')
    return eval_tokenizer.batch_decode(res, skip_special_tokens=True)

# Flatten the list of lists into a single list of documents
documents_flattened = [doc for sublist in documents for doc in sublist]

# Initialize text splitter
chunk_size = 400  # Recommended chunk size
chunk_overlap = 100  # Recommended overlap
text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

def split_document(document):
    chunks = text_splitter.split_text(document.page_content)
    return [Document(page_content=chunk) for chunk in chunks]

chunked_documents = []
for doc in documents_flattened:
    chunked_documents.extend(split_document(doc))

chunked_texts = [doc.page_content for doc in chunked_documents]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunked_texts, convert_to_tensor=True)
print("embedding_done")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.cpu().numpy())

def retrieve_top_k(question, k=3):
    question_embedding = model.encode([question])
    distances, indices = index.search(question_embedding, k)
    return [chunked_texts[idx] for idx in indices[0]]

def baseline_rag(question):
    retrieved_documents = retrieve_top_k(question)
    context = " ".join(retrieved_documents)
    input_text = f"Answer the question with given context: {question}\nContext: {context}\nAnswer:"
    generated = gen(base_model, input_text, 100)
    # Print the full generated output for debugging
    print("Full generated output:", generated)

    # Get the generated text and remove the input_text part to isolate the answer
    full_generated_text = generated[0]
    answer = full_generated_text.replace(input_text, '').strip()

    # Further process to extract the answer part only, if needed
    answer_lines = answer.split('\n')
    if len(answer_lines) > 1:
        answer = answer_lines[0].strip()  # Typically, the first line after input_text is the answer

    # Fallback if answer is still incorrectly extracted
    if not answer or "Question" in answer:
        answer = "Answer not found or incorrect extraction"

    print("Extracted Answer:", answer)
    return answer, context

# Load the CSV file
csv_file = 'path/to/your/csvfile.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Add columns for the generated answers and retrieved chunks
df['Generated Answer'] = ''
df['Retrieved Chunks'] = ''

# Iterate through the questions in the CSV file and generate answers
for index, row in df.iterrows():
    question = row['Question']
    expected_chunks = row['Expected Chunks']
    generated_answer, retrieved_chunks = baseline_rag(question)
    
    df.at[index, 'Generated Answer'] = generated_answer
    df.at[index, 'Retrieved Chunks'] = retrieved_chunks

# Save the updated DataFrame to a new CSV file
df.to_csv('updated_finetuneRAGresults.csv', index=False)

print("Results saved to updated_finetuneRAGresults.csv")
