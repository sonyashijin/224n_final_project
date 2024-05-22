import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from functools import partial

# Define preprocessing functions
def preprocess_batch(batch, tokenizer, max_length):
    """Tokenizing a batch"""
    return tokenizer(
        batch["truncated_combined_text"],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    :param seed (int): Random seed for shuffling
    :param dataset: The dataset to preprocess
    """
    print("Preprocessing dataset...")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context'],
    )
    dataset = dataset.shuffle(seed=seed)
    return dataset

# Load the preprocessed datasets
train_dataset = load_from_disk("~/final_project/processed_data/processed_train_dataset")
eval_dataset = load_from_disk("~/final_project/processed_data/processed_validation_dataset")

# Load the base model and tokenizer
model_name = 'microsoft/phi-2'
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
device_map = {"": 0}

original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left", add_eos_token=True, add_bos_token=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Print the number of trainable parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))

# Preprocess the datasets
max_length = 2048  # Adjust based on your model and dataset
seed = 42
train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset)

# PEFT Configuration
config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# Enable gradient checkpointing to reduce memory usage during fine-tuning
original_model.gradient_checkpointing_enable()

# Prepare the model for k-bit training
original_model = prepare_model_for_kbit_training(original_model)

# Get the PEFT model
peft_model = get_peft_model(original_model, config)

# Define training arguments
output_dir = './peft-training/final-checkpoint'
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=500,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=25,
    logging_dir="./logs",
    save_strategy="steps",
    save_steps=25,
    evaluation_strategy="steps",
    eval_steps=25,
    do_eval=False,
    gradient_checkpointing=True,
    report_to="none",
    overwrite_output_dir=True,
    group_by_length=True,
)

# Disable cache for PEFT model
peft_model.config.use_cache = False

# Initialize the Trainer with additional logging
class CustomTrainer(Trainer):
    def on_log(self, logs: dict, **kwargs):
        super().on_log(logs, **kwargs)
        if 'epoch' in logs:
            print(f"Epoch {logs['epoch']}")
        if 'step' in logs:
            print(f"Step {logs['step']}")
        if 'loss' in logs:
            print(f"Loss: {logs['loss']}")

# Initialize the Trainer
peft_trainer = CustomTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=peft_training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training with logging
print("Starting training...")
peft_trainer.train()

# Free memory by deleting model and trainer
del original_model
del peft_trainer
torch.cuda.empty_cache()

print("Training complete and memory freed.")
