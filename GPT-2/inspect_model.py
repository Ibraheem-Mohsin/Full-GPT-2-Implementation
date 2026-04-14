import torch
import tiktoken
from model import GPT, GPTConfig
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")


print("Building model architecture...")
config = GPTConfig(vocab_size=50304)
model = GPT(config) 

print("Loading trained weights...")
model.load_state_dict(torch.load("trained_model.pt", map_location=device))

model.to(device)
model.eval()  # Turn off training mode

# one Shakespeare sentence
text = "He's one honest enough"
tokens = enc.encode(text)
idx = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

print(f"\nInput text: '{text}'")
print(f"Token IDs: {tokens}")
print(f"Sequence Length: {idx.size(1)} tokens\n")

# Pass it through the model
with torch.no_grad():
    logits, _ = model(idx)

first_token_id = idx[0, 0].item()
first_word = enc.decode([first_token_id])

# Print the Token Embeddings
print("=== TOKEN EMBEDDINGS ===")
print(f"Shape: {model.diagnostic_tok_emb.shape}")
print(f"Here is the 768-dimensional token embedding for the first word '{first_word}':")
print(model.diagnostic_tok_emb[0, 0, :10], "... (showing first 10 of 768 numbers)\n")



# ---------------------------------------------------------

# store the 5 next predicted words here
generated_ids = []

for i in range(5):
    # Get the model's predictions for the current sequence
    with torch.no_grad():
        logits, _ = model(idx)
    
    # Isolate the scores for the very last word in the sequence
    logits = logits[:, -1, :] # Shape: (Batch, Vocab_Size)
    
    # Top-K filter (Keep top 50, destroy the rest)
    top_values, _ = torch.topk(logits, 50)
    # Find the 50th highest value
    cutoff_value = top_values[:, [-1]] 
    # Any score lower than the cutoff gets set to negative infinity
    logits[logits < cutoff_value] = -float('Inf')
    
    
    # Because the words below 50 are -Inf, they become exactly 0%
    probs = F.softmax(logits, dim=-1)
    
    # Sample 1 token from our Top 50 distribution
    next_id_tensor = torch.multinomial(probs, num_samples=1)
    
    # Save the ID and append it to our sequence for the next loop
    generated_ids.append(next_id_tensor.item())
    idx = torch.cat((idx, next_id_tensor), dim=1)

# Translate the 5 new IDs back into English
predicted_text = enc.decode(generated_ids)

print(f"The next 5 predicted words are: '{predicted_text}' ")















