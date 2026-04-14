import torch
import time
import sys
import math
from model import GPT, GPTConfig
from data import DataLoaderLite


# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

print(f"using device: {device}")

# for reproducibility (weights)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=32)  # each of our rows is 32 tokens
#sgd vs bgd vs mini-batch gradient descent
#gradient convergence, local minima/global minima. Weight decay, momentum, normalization.

torch.set_float32_matmul_precision("high")  # tensorfloat 32

## STEP 7 in video
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 8
max_steps = 42


def get_lr(it): # cosine lr scheduler
   #1) linear warmup for warmup_iters steps 
    if it < warmup_steps:                       # this is the warmup region
        return max_lr * (it+1) / warmup_steps
    #2) if it › lr_decay_iters, return min learning rate 
    if it > max_steps:
        return min_lr
    #3) in between, use cosine decay down to decrease learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps) #region in between where we calculate the cosine learning rate schedule
    assert 0 <= decay_ratio <= 1. 
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
for step in range (max_steps):
    t0 = time.time()
    x,y = train_loader.next_batch() 
    x,y = x.to(device), y.to(device) #move tensors from CPU to GPU (for when people have access to GPU's)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # only possible on Nvidia Ampere and later GPU series
        logits,loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient norm clipping
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {step:4d} | loss : {loss.item():.6f} |norm: {norm:.4f} |  dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec}")

    if loss.item() <= 6.44:
        print(f"\n (Actual loss: {loss.item():.4f})!")
        torch.save(model.state_dict(), '6.44model.pt')
        break 

sys.exit(0)


# STEPS 5 & 6 in video ( TF32, BF-16, GPT-3 Hyperparamters, nice/ugly numbers)
# train_loader = DataLoaderLite(B=4, T=32)

# torch.set_float32_matmul_precision('high') #tensorfloat 32

# # get logits
# model = GPT(GPTConfig(vocab_size=50304)) #the default config uses 124M parameters
# model.to(device)
# #model = torch.compile(model) # will take out the python interpreter from the forward pass entirely (hence knows everything it needs to run (can optimize it cuz it doesn't have to do round trips again and again to/from memory) )

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# for i in range (20):
#     t0 = time.time()
#     x,y = train_loader.next_batch()
#     x,y = x.to(device), y.to(device) #move tensors from CPU to GPU (for when people have access to GPU's)
#     optimizer.zero_grad()
#     with torch.autocast(device_type=device, dtype=torch.bfloat16): # only possible on Nvidia Ampere and onwards GPU series
#         logits,loss = model(x, y)
#     loss.backward()
#     norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient norm clipping
#     optimizer.step()

#     t1 = time.time()
#     dt = t1-t0 # time difference in seconds
#     tokens_processed = train_loader.B * train_loader.T
#     tokens_per_sec = tokens_processed / dt
#     print(f"step {i:4d} | loss : {loss.item():.6f} |norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec}")


# sys.exit(0)

## STEP 1 in video

# model = GPT.from_pretrained ('gpt2') #initialized the model with GPT-2 weights from tiktoken OpenAI Library
# print ("didn't crash yay!")


# get logits
# model = GPT(GPTConfig()) #creating the default config uses 124M parameters
# model.to(device)


## STEP 3

# logits, loss = model(x,y)
# print(loss) ## will print loss at initialization (ideally it should be 1/50257)
# sys.exit(0)


## STEP 4

## optimizing model for 1 batch after printing loss

## get a data batch


# enc = tiktoken.get_encoding('gpt2')
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch. tensor(tokens [:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for i in range (20):
#     optimizer.zero_grad()
#     logits, loss = model(x, y)
#     loss.backward()
#     optimizer.step()
#     print(f"step {i}, loss: {loss.item()}")

sys.exit(0)


# STEP 2

## prefix tokens

# model.eval()
# num_return_sequences = 5
# max_length = 30
# enc = tiktoken.get_encoding ('gpt2')
# tokens = enc.encode("Hello, I'm a language model, ")
# print("Token IDs:", tokens)
# print(f"Count: {len(tokens)}")

# print("\n--- Visual Breakdown ---")
# for i, j in enumerate(tokens):
#     # We decode each ID back to a string to see the chunk
#     chunk = enc.decode([j])
#     print(f"Token {i+1}: {j} -> '{chunk}'")

# tokens = torch.tensor(tokens, dtype=torch. long) # (9, ) #converts python list into Pytorch tensor
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 9)
# x = tokens.to(device)

# # generating some samples! Right now x is (B, T) where B = 5, T = 9
# # set the seed to 42
# torch.manual_seed (42)
# torch.cuda.manual_seed (42)
# while x.size(1) < max_length: #second dimension (T)
# # forward the model to get the logits
#     with torch.no_grad():
#         logits, _ = model(x) # (B, T, vocab_size)
#         # take the logits at the last position (we only care about last column's logits)
#         logits = logits[:, -1, :] # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50). Hugging Face default is 50. And this ensures that we are not sampling rare tokens. We are always sampling tokens that are in the top 50 of most likely tokens. This helps keep the model on track so it doesn't blabber on and get lost easily.
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices. We get this new column of tokens and append it to x
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode (tokens)
#     print (">", decoded)
