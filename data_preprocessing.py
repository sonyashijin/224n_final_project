import random
import torch
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Login to Huggingface
from huggingface_hub import interpreter_login
interpreter_login()

# Load the dataset
huggingface_dataset_name = "hotpot_qa"
dataset = load_dataset(huggingface_dataset_name, "distractor")

print("Dataset loaded successfully")

# Configure model and tokenizer
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
device_map = {"": 0}

model_name = 'microsoft/phi-2'
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)

print("Model loaded successfully")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left", add_eos_token=True, add_bos_token=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

eval_tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

MAX_LENGTH = get_max_length(original_model)

def preprocess_batch(batch, tokenizer, max_length):
    """Tokenizing a batch"""
    return tokenizer(
        batch["truncated_combined_text"],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

def build_context(data_point):
    supporting_titles = data_point['supporting_facts']['title']
    supporting_sent_ids = data_point['supporting_facts']['sent_id']

    relevant_context = []
    other_context = []

    for i, title in enumerate(data_point['context']['title']):
        if title in supporting_titles:
            relevant_sentences = [
                sentence for j, sentence in enumerate(data_point['context']['sentences'][i])
                if title in supporting_titles and j in supporting_sent_ids
            ]
            relevant_context.extend(relevant_sentences)
        else:
            other_context.extend(data_point['context']['sentences'][i])

    # Randomly sample the remaining context to pad the relevant context
    random.shuffle(other_context)
    return relevant_context, other_context

def truncate_context(data_point, max_length=2048):
    relevant_context, other_context = build_context(data_point)

    instruction_prompt = "### Instruct: With the given context, please answer the question in one word."
    question = data_point['question']
    answer = data_point['answer']
    context_key = "Context:"
    response_key = "### Output:"

    # Calculate the length of static parts
    static_parts = f"{instruction_prompt}\nQuestion: {question}\n{context_key}\n{response_key}\n{answer}"
    static_parts_length = len(tokenizer.tokenize(static_parts))

    # Calculate the available length for the context
    available_length = max_length - static_parts_length

    # Ensure available_length is non-negative
    if available_length <= 0:
        raise ValueError("Static parts of the prompt exceed the max length.")

    # Tokenize relevant context and check length
    relevant_context_str = ' '.join(relevant_context)
    relevant_tokens = tokenizer(relevant_context_str, return_tensors='pt').input_ids[0]

    if len(relevant_tokens) > available_length:
        # Truncate relevant context if it exceeds available length
        truncated_relevant_tokens = relevant_tokens[:available_length]
        truncated_relevant_context = tokenizer.decode(truncated_relevant_tokens, skip_special_tokens=True)
        truncated_combined_text = f"{instruction_prompt}\nQuestion: {question}\n{context_key}\n{truncated_relevant_context}\n{response_key}\n{answer}"
    else:
        # If relevant context fits, add as much irrelevant context as possible
        truncated_relevant_context = relevant_context_str
        remaining_length = available_length - len(relevant_tokens)
        irrelevant_context_str = ' '.join(other_context)
        irrelevant_tokens = tokenizer(irrelevant_context_str, return_tensors='pt', truncation=True, max_length=remaining_length).input_ids[0]
        truncated_irrelevant_context = tokenizer.decode(irrelevant_tokens, skip_special_tokens=True)

        # Combine relevant and truncated irrelevant context
        combined_context = truncated_relevant_context + ' ' + truncated_irrelevant_context
        combined_context_list = combined_context.split('. ')
        random.shuffle(combined_context_list)
        shuffled_combined_context = '.'.join(combined_context_list)

        truncated_combined_text = f"{instruction_prompt}\nQuestion: {question}\n{context_key}\n{shuffled_combined_context}\n{response_key}\n{answer}"

    final_tokens = tokenizer(truncated_combined_text, return_tensors='pt').input_ids[0]
    if len(final_tokens) > max_length:
        truncated_combined_text = tokenizer.decode(final_tokens[:max_length-2], skip_special_tokens=True)

    # Update the data_point with the truncated context
    data_point['truncated_combined_text'] = truncated_combined_text
    return data_point

def process_dataset(dataset, max_length=MAX_LENGTH):
    processed_dataset = dataset.map(lambda x: truncate_context(x, max_length), batched=False)
    return processed_dataset

def check_lengths(dataset, max_length=MAX_LENGTH):
    for data_point in dataset:
        tokens = tokenizer(data_point['truncated_combined_text'], return_tensors='pt')
        if tokens.input_ids.shape[1] > max_length:
            print(f"Sequence length {tokens.input_ids.shape[1]} exceeds max length {max_length}")

# Main function
if __name__ == "__main__":
    # Process the train and validation datasets
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']

    # Sample 20,000 rows from the train split and process
    sampled_train_dataset = train_dataset.shuffle(seed=42).select(range(20000))

    print("Processing train dataset...")
    sampled_train_dataset = process_dataset(sampled_train_dataset)
    print("Processing validation dataset...")
    validation_dataset = process_dataset(validation_dataset)

    # Check the lengths after processing
    print("Checking lengths of train dataset...")
    check_lengths(sampled_train_dataset)
    print("Checking lengths of validation dataset...")
    check_lengths(validation_dataset)

    # Save the processed datasets
    sampled_train_dataset.save_to_disk("processed_train_dataset")
    validation_dataset.save_to_disk("processed_validation_dataset")

    print("Preprocessing and saving complete.")
