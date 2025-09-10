import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Configuration ---
MODEL_NAME = "distilgpt2"
DATASET_PATH = "groundwater.jsonl"
OUTPUT_DIR = "./groundwater-llm"

# --- 1. Load Dataset ---
def load_and_prepare_dataset(tokenizer):
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    # We need to format the data into a single text column for the trainer
    def format_prompt(example):
        return {'text': f'Prompt: {example["prompt"]}\nCompletion:{example["completion"]}'}
    
    dataset = dataset.map(format_prompt)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

# --- 2. Load Model and Tokenizer ---
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True, # Use 8-bit quantization to save memory
        device_map='auto', # Automatically distribute model across available devices
    )
    return model, tokenizer

# --- 3. Configure LoRA ---
def configure_lora(model):
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16, # Rank of the update matrices
        lora_alpha=32, # Alpha parameter for scaling
        target_modules=["c_attn", "c_proj", "c_fc"], # Target all linear layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get the PEFT model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model

# --- 4. Train the Model ---
def train_model(model, tokenizer, dataset):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the fine-tuned model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully.")

if __name__ == "__main__":
    # Execute the pipeline
    model, tokenizer = load_model_and_tokenizer()
    tokenized_data = load_and_prepare_dataset(tokenizer)
    peft_model = configure_lora(model)
    train_model(peft_model, tokenizer, tokenized_data)
