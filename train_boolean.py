#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import json
from pathlib import Path
import random

# Hyperparameters (Recommended from spec.md)
batch_size_default = 64
block_size_default = 64 # Tuned for expression length
max_iters_default = 5000 # Increased for convergence
eval_interval_default = 500
learning_rate_default = 3e-4
n_embd_default = 128
n_head_default = 4
n_layer_default = 4
dropout_default = 0.1 # 0.1-0.2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200

# ------------

print(f"Using device: {device}")

torch.manual_seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boolean_train.txt')
parser.add_argument('--val_dataset', type=str, default='boolean_val.txt')
parser.add_argument('--model_name', type=str, default='model_weights_part2.pth')
parser.add_argument('--max_iters', type=int, default=max_iters_default)
parser.add_argument('--batch_size', type=int, default=batch_size_default)
parser.add_argument('--block_size', type=int, default=block_size_default)
parser.add_argument('--eval_interval', type=int, default=eval_interval_default)
parser.add_argument('--learning_rate', type=float, default=learning_rate_default)
parser.add_argument('--n_embd', type=int, default=n_embd_default)
parser.add_argument('--n_head', type=int, default=n_head_default)
parser.add_argument('--n_layer', type=int, default=n_layer_default)
parser.add_argument('--dropout', type=float, default=dropout_default)
args = parser.parse_args()

batch_size = args.batch_size
block_size = args.block_size
max_iters = args.max_iters
eval_interval = args.eval_interval
learning_rate = args.learning_rate
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
train_path = DATASET_DIR / args.dataset
val_path = DATASET_DIR / args.val_dataset

# Tokenizer
# Tokens: True, False, AND, OR, NOT, XOR, (, ), =, \n
# We need a custom tokenizer, not char-level
def get_tokens(text):
    # Pad symbols with spaces to ensure clean split
    text = text.replace('(', ' ( ').replace(')', ' ) ').replace('=', ' = ').replace('\n', ' \n ')
    return [t for t in text.split(' ') if t]

# Build vocabulary from training data
print(f"Loading training data from {train_path}")
with open(train_path, 'r', encoding='utf-8') as f:
    train_text = f.read()

tokens = get_tokens(train_text)
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary: {vocab}")

stoi = { ch:i for i,ch in enumerate(vocab) }
itos = { i:ch for i,ch in enumerate(vocab) }

def encode(s):
    # s is a string (line or full text)
    # We need to tokenize it first
    t = get_tokens(s)
    return [stoi[c] for c in t]

def decode(l):
    return ' '.join([itos[i] for i in l])

# Prepare data tensors
train_data = torch.tensor(encode(train_text), dtype=torch.long)

if val_path.exists():
    print(f"Loading validation data from {val_path}")
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    val_data = torch.tensor(encode(val_text), dtype=torch.long)
else:
    print("Validation file not found, splitting training data.")
    n = int(0.9*len(train_data))
    val_data = train_data[n:]
    train_data = train_data[:n]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model Architecture (Same as gpt.py but with updated vocab/hyperparams)
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        v = self.value(x) 
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Ensure idx is within block_size
        if T > block_size:
             idx = idx[:, -block_size:]
             T = block_size
             if targets is not None:
                 targets = targets[:, -block_size:]

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        x = self.blocks(x) 
        x = self.ln_f(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

model = GPTLanguageModel()
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save model
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / args.model_name

print(f"Saving model to {model_path}")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'vocab': vocab, # Save vocab list for tokenizer reconstruction
    'hyperparameters': {
        'batch_size': batch_size,
        'block_size': block_size,
        'max_iters': max_iters,
        'learning_rate': learning_rate,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'dropout': dropout
    }
}, model_path)
