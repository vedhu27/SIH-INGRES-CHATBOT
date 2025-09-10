import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Configuration ---
MODEL_PATH = "./groundwater-llm"

# --- Load Model and Tokenizer ---
def load_finetuned_model():
    """Loads the fine-tuned model and tokenizer from the specified path."""
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load the base model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", # Make sure this matches the base model used in training
        load_in_8bit=True,
        device_map='auto',
    )
    
    # Load the LoRA model
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval() # Set the model to evaluation mode
    
    print("Model loaded successfully.")
    return model, tokenizer

# --- Generate Response ---
def generate_response(model, tokenizer, prompt):
    """Generates a response from the model given a prompt."""
    # Format the input prompt
    input_text = f"Prompt: {prompt}\nCompletion:"
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50, # Limit the length of the generated response
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the completion part
    completion = response.split("Completion:")[-1].strip()
    return completion

# --- Main Chatbot Loop ---
def run_chatbot(model, tokenizer):
    """Runs the main command-line interface for the chatbot."""
    print("\n--- Groundwater Chatbot CLI ---")
    print("Type your question and press Enter. Type 'exit' to quit.")
    
    while True:
        user_prompt = input("\nYou: ")
        if user_prompt.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        
        # Generate and print the model's response
        model_response = generate_response(model, tokenizer, user_prompt)
        print(f"Bot: {model_response}")

if __name__ == "__main__":
    # Load the model and run the chatbot
    finetuned_model, tokenizer = load_finetuned_model()
    run_chatbot(finetuned_model, tokenizer)
