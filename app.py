import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('models/my_tokenizer')
model = GPT2LMHeadModel.from_pretrained('models/saved_model')

def generate_text(input_text, max_length=100, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.0):
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=model.device)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            attention_mask=attention_mask,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated output text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Replace line breaks with a custom character for preservation
    generated_text = generated_text.replace("\n", "<br>")

    return generated_text

def generate_text_with_typing(input_text, max_length=100, temperature=1.0, top_k=50, top_p=1, repetition_penalty=1.0):
    generated_text = generate_text(input_text, max_length, temperature, top_k, top_p, repetition_penalty)
    tokens = generated_text.split()

    # Create an event stream for the typing effect
    for i, token in enumerate(tokens, 1):
        yield " ".join(tokens[:i])