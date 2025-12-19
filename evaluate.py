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
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
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

def evaluate_model(model, stoi, itos, num_samples=500, verbose=False, advanced=False):
    """
    Generates random math problems and checks if the model solves them correctly.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Define the problem space
    operators = ['+', '-', '*', '/']
    max_val = 99 if advanced else 9
    
    print(f"\nEvaluating on {num_samples} random examples (Advanced: {advanced})...")
    print("-" * 40)
    
    for _ in range(num_samples):
        op = random.choice(operators)

        if op == '/':
            while True:
                a = random.randint(0, max_val)
                b = random.randint(1, max_val)
                if a % b == 0:
                    res = a // b
                    break
        else:
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)

            if op == '+':
                res = a + b
            elif op == '-':
                res = a - b
            elif op == '*':
                res = a * b
            
        prompt = f"{a}{op}{b}="
        expected = str(res)
        
        # Check if characters are in vocab
        if not all(c in stoi for c in prompt):
            continue
            
        # Encode
        input_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        generated_ids = model.generate(input_ids, max_new_tokens=len(expected) + 1)
        generated_text = ''.join([itos[i] for i in generated_ids[0].tolist()])
        
        # Extract the answer part
        answer_part = generated_text[len(prompt):]
        
        predicted = answer_part.split('\n')[0].strip()
        
        # Strict equality check
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        if verbose:
            print(f"Q: {prompt} | Predicted: {predicted} | Expected: {expected} | {'Correct' if is_correct else 'Incorrect'}")
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print("-" * 40)
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

def main():
    global batch_size, block_size, n_embd, n_head, n_layer, dropout, vocab_size
    
    parser = argparse.ArgumentParser(description='Evaluate GPT Math Model')
    parser.add_argument('--model', type=str, default='gpt_math_model.pth', help='Name of the model file in models/ directory')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=500, help='Number of evaluation samples to run')
    parser.add_argument('--advanced', action='store_true', help='Evaluate on advanced math problems (0-99)')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    # MODELS_DIR = BASE_DIR / "models"
    model_path = BASE_DIR / args.model

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load hyperparameters
    hyperparams = checkpoint['hyperparameters']
    batch_size = hyperparams['batch_size']
    block_size = hyperparams['block_size']
    n_embd = hyperparams['n_embd']
    n_head = hyperparams['n_head']
    n_layer = hyperparams['n_layer']
    dropout = hyperparams['dropout']
    
    # Load mappings
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    chars = checkpoint['chars']
    vocab_size = len(chars)
    
    print(f"Model config: n_embd={n_embd}, n_layer={n_layer}, n_head={n_head}, vocab_size={vocab_size}")

    # Initialize model
    model = GPTLanguageModel()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Run evaluation
    evaluate_model(model, stoi, itos, verbose=args.verbose, num_samples=args.iterations, advanced=args.advanced)

if __name__ == "__main__":
    main()