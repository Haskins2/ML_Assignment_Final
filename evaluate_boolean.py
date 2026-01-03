#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from pathlib import Path
import random

# Globals needed for the model classes (will be set from checkpoint)
batch_size = None
block_size = None
n_embd = None
n_head = None
n_layer = None
dropout = None
vocab_size = None
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model Architecture (Must match train_boolean.py)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > block_size:
             idx = idx[:, -block_size:]
             T = block_size
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_tokens(text):
    text = text.replace('(', ' ( ').replace(')', ' ) ').replace('=', ' = ').replace('\n', ' \n ')
    return [t for t in text.split(' ') if t]

def evaluate_model(model, stoi, itos, dataset_path, num_samples=500, verbose=False):
    model.eval()
    correct = 0
    total = 0
    
    print(f"Evaluating on {dataset_path}...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Shuffle and select samples
    random.shuffle(lines)
    samples = lines[:num_samples]
    
    for line in samples:
        line = line.strip()
        if not line: continue
        
        # Split into prompt and expected
        # Format: <expr> = <bool>
        if '=' not in line: continue
        
        parts = line.split('=')
        expr = parts[0].strip()
        expected = parts[1].strip()
        
        prompt = f"{expr} ="
        
        # Tokenize prompt
        prompt_tokens = get_tokens(prompt)
        
        # Check vocab
        if not all(t in stoi for t in prompt_tokens):
            if verbose: print(f"Skipping {prompt}: unknown tokens")
            continue
            
        input_ids = torch.tensor([stoi[t] for t in prompt_tokens], dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate 1 token (True or False)
        # We expect the next token to be the answer
        with torch.no_grad():
            # We can just look at the logits for the next token
            logits, _ = model(input_ids)
            # Get the last token's logits
            next_token_logits = logits[0, -1, :]
            # Greedy decode
            predicted_id = torch.argmax(next_token_logits).item()
            predicted = itos[predicted_id]
            
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            
        if verbose:
            print(f"Q: {prompt} | Pred: {predicted} | Exp: {expected} | {'✓' if is_correct else '✗'}")
            
        total += 1
        
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def main():
    global batch_size, block_size, n_embd, n_head, n_layer, dropout, vocab_size
    
    parser = argparse.ArgumentParser(description='Evaluate Boolean GPT Model')
    parser.add_argument('--model', type=str, default='model_weights_part2.pth', help='Model file')
    parser.add_argument('--dataset', type=str, default='boolean_test.txt', help='Test dataset')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    DATASET_DIR = BASE_DIR / "datasets"
    
    model_path = MODELS_DIR / args.model
    dataset_path = DATASET_DIR / args.dataset

    if not model_path.exists():
        print(f"Error: Model {model_path} not found")
        return

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    hyperparams = checkpoint['hyperparameters']
    batch_size = hyperparams['batch_size']
    block_size = hyperparams['block_size']
    n_embd = hyperparams['n_embd']
    n_head = hyperparams['n_head']
    n_layer = hyperparams['n_layer']
    dropout = hyperparams['dropout']
    
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab = checkpoint.get('vocab', []) # Fallback if not saved
    vocab_size = len(stoi)
    
    print(f"Config: n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, vocab={vocab_size}")

    model = GPTLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    evaluate_model(model, stoi, itos, dataset_path, verbose=args.verbose)

if __name__ == "__main__":
    main()
