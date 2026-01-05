#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
from pathlib import Path
import random
import re

# Globals needed for the model classes (will be set from checkpoint)
batch_size = None
block_size = None
n_embd = None
n_head = None
n_layer = None
dropout = None
vocab_size = None
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def eval_bool_expr(expr):
    # Convert dataset format to Python syntax for evaluation
    # AND -> and
    # OR -> or
    # NOT -> not
    # XOR -> ^ (bitwise XOR works as logical XOR for booleans)
    py_expr = expr.replace("XOR", "^").replace("AND", "and").replace("OR", "or").replace("NOT", "not")
    return eval(py_expr)

def generate_boolean_problem(advanced=False):
    ops = ['AND', 'OR', 'XOR']
    booleans = ['True', 'False']

    if advanced:
        # Mix of simple and complex
        if random.random() < 0.3:
             return generate_boolean_problem(advanced=False)
        
        a = random.choice(booleans)
        b = random.choice(booleans)
        c = random.choice(booleans)
        
        op1 = random.choice(ops)
        op2 = random.choice(ops)
        
        # Patterns matching generate_boolean_dataset.py
        pattern = random.choice([1, 2, 3, 4, 5])
        
        if pattern == 1:
            expr = f"{a} {op1} {b} {op2} {c}"
        elif pattern == 2:
            expr = f"({a} {op1} {b}) {op2} {c}"
        elif pattern == 3:
            expr = f"{a} {op1} ({b} {op2} {c})"
        elif pattern == 4:
             expr = f"NOT {a} {op1} {b}"
        else:
             expr = f"{a} {op1} NOT {b}"
             
        try:
            res = eval_bool_expr(expr)
            return f"{expr}=", str(res), 'complex_bool'
        except:
            return generate_boolean_problem(advanced=True) # Retry

    if random.random() < 0.2: # 20% chance for NOT
        val = random.choice(booleans)
        expr = f"NOT {val}"
        res = not (val == 'True')
        op = 'NOT'
    else:
        op = random.choice(ops)
        val1 = random.choice(booleans)
        val2 = random.choice(booleans)
        expr = f"{val1} {op} {val2}"
        
        v1 = val1 == 'True'
        v2 = val2 == 'True'
        
        if op == 'AND':
            res = v1 and v2
        elif op == 'OR':
            res = v1 or v2
        elif op == 'XOR':
            res = v1 != v2
            
    return expr + "=", str(res), op

def generate_math_problem(advanced=False):
    operators = ['+', '-', '*', '/']
    max_val = 99 if advanced else 9
    
    # 30% chance for 3-input BIMDAS
    if random.random() < 0.3:
            ops = ['+', '-', '*']
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            c = random.randint(0, max_val)
            
            op1 = random.choice(ops)
            op2 = random.choice(ops)
            pattern = random.choice([1, 2, 3])
            
            if pattern == 1:
                expr = f"{a}{op1}{b}{op2}{c}"
            elif pattern == 2:
                expr = f"({a}{op1}{b}){op2}{c}"
            else:
                expr = f"{a}{op1}({b}{op2}{c})"
            
            try:
                res = eval(expr)
                return f"{expr}=", str(res), 'bimdas'
            except:
                return generate_math_problem(advanced) # Retry
    else:
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
            
        return f"{a}{op}{b}=", str(res), op

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
    Generates random problems and checks if the model solves them correctly.
    """
    model.eval()
    
    # Metrics
    correct = 0
    total = 0
    op_stats = {} # {op: {'correct': 0, 'total': 0}}
    total_cer = 0
    total_mae = 0
    total_mse = 0
    
    # Detect mode
    is_boolean = 'True' in stoi
    print(f"\nEvaluating on {num_samples} random examples (Mode: {'Boolean' if is_boolean else 'Math'}, Advanced: {advanced})...")
    print("-" * 60)
    
    for _ in range(num_samples):
        if is_boolean:
            prompt, expected, op = generate_boolean_problem(advanced)
        else:
            prompt, expected, op = generate_math_problem(advanced)
        
        # Initialize op stats if needed
        if op not in op_stats:
            op_stats[op] = {'correct': 0, 'total': 0}
            
        # Tokenize
        if is_boolean:
            # Use regex tokenizer for boolean
            tokens = re.findall(r'\w+|[=]|\n| ', prompt)
            if not all(t in stoi for t in tokens):
                continue
            input_ids = torch.tensor([stoi[t] for t in tokens], dtype=torch.long, device=device).unsqueeze(0)
        else:
            # Use char tokenizer for math
            if not all(c in stoi for c in prompt):
                continue
            input_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        # For boolean, we expect 1 token answer (True/False) usually, but let's allow a few
        max_new = 1 if is_boolean else len(expected) + 1
        generated_ids = model.generate(input_ids, max_new_tokens=max_new)
        
        if is_boolean:
             # Decode using list of tokens
             generated_tokens = [itos[i] for i in generated_ids[0].tolist()]
             # Reconstruct string (simple join for now, though spaces might be tricky if not handled in tokenizer)
             # In gpt.py, decode is ''.join([itos[i]...])
             generated_text = ''.join(generated_tokens)
        else:
             generated_text = ''.join([itos[i] for i in generated_ids[0].tolist()])
        
        # Extract answer
        # The prompt is part of generated_text
        answer_part = generated_text[len(prompt):]
        
        # Clean up answer
        if is_boolean:
            # For boolean, we just want the first token that looks like True/False
            # But our tokenizer includes spaces.
            # Let's just take the first non-space token or the whole thing stripped
            predicted = answer_part.strip()
            # Sometimes it might generate "True " or "True\n"
            match = re.search(r'(True|False)', predicted)
            if match:
                predicted = match.group(1)
        else:
            predicted = answer_part.split('\n')[0].strip()
        
        # 1. Exact Match
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            op_stats[op]['correct'] += 1
        op_stats[op]['total'] += 1
        
        # 3. CER
        cer = levenshtein_distance(predicted, expected) / max(len(expected), 1)
        total_cer += cer
        
        # 4 & 5. MAE / MSE (Math only)
        if not is_boolean:
            try:
                pred_val = float(predicted)
                exp_val = float(expected)
                error = abs(pred_val - exp_val)
                total_mae += error
                total_mse += error ** 2
            except ValueError:
                # If prediction is not a number, treat as a large error or skip?
                # Usually we might assign a penalty or just skip. 
                # For this assignment, let's skip numerical metrics for invalid outputs
                pass
        
        if verbose:
            print(f"Q: {prompt} | Pred: {predicted} | Exp: {expected} | {'✓' if is_correct else '✗'}")
        total += 1

    # Calculate aggregates
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_cer = total_cer / total if total > 0 else 0
    avg_mae = total_mae / total if total > 0 and not is_boolean else 0
    avg_mse = total_mse / total if total > 0 and not is_boolean else 0

    print("-" * 60)
    print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Character Error Rate (CER): {avg_cer:.4f}")
    
    if not is_boolean:
        print(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
        print(f"Mean Squared Error (MSE): {avg_mse:.4f}")
    
    print("\nOperation-wise Accuracy:")
    for op, stats in op_stats.items():
        op_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {op}: {op_acc:.2f}% ({stats['correct']}/{stats['total']})")
        
    return accuracy

def main():
    global batch_size, block_size, n_embd, n_head, n_layer, dropout, vocab_size
    
    parser = argparse.ArgumentParser(description='Evaluate GPT Math Model')
    parser.add_argument('--model', type=str, default='gpt_math_model.pth', help='Name of the model file in models/ directory')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=500, help='Number of evaluation samples to run')
    parser.add_argument('--advanced', action='store_true', help='Evaluate on advanced problems (math: 0-99, boolean: complex expressions)')
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    model_path = MODELS_DIR / args.model

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