import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Configuration ---
MODEL_PATH = "./groundwater-llm"

# --- Load Model and Tokenizer (similar to inference.py) ---
def load_finetuned_model():
    """Loads the fine-tuned model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        load_in_8bit=True,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    return model, tokenizer

# Load the model and tokenizer once when the app starts
model, tokenizer = load_finetuned_model()
print("Model loaded for Gradio app.")

# --- Prediction Function ---
def chatbot_response(prompt):
    """Generates a response for the Gradio interface."""
    input_text = f"Prompt: {prompt}\nCompletion:"
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = response.split("Completion:")[-1].strip()
    return completion

# --- Gradio Interface ---
def create_gradio_app():
    """Creates and launches the Gradio web application."""
    iface = gr.Interface(
        fn=chatbot_response,
        inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
        outputs="text",
        title="🌍 Groundwater Chatbot",
        description="Ask me about groundwater data (2024-25) in South India.",
        allow_flagging="never",
        theme=gr.themes.Soft()
    )
    
    print("Launching Gradio app...")
    iface.launch()

if __name__ == "__main__":
    create_gradio_app()
