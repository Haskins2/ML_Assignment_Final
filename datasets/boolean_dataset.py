#!/usr/bin/env python3

import argparse
from pathlib import Path
import random

# argparse practice
parser = argparse.ArgumentParser(description="Generate a boolean logic dataset.")
parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth of the expression tree.")
parser.add_argument("--count", type=int, default=10000, help="Number of samples to generate.")
parser.add_argument("--ood_depth", action="store_true", help="Generate OOD depth samples.")
parser.add_argument("--ood_length", action="store_true", help="Generate OOD length samples.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "ood_depth", "ood_length"], help="Dataset split.")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_NAME = f"boolean_{args.split}.txt"
OUTPUT_FILE = DATASET_DIR / DATASET_NAME

OPS = ['AND', 'OR', 'XOR']

def generate_expression(depth):
    if depth == 0:
        return random.choice(['True', 'False'])
    
    # Choose between atom, unary op, or binary op
    # Bias towards binary ops to build complexity
    choice = random.random()
    
    if choice < 0.2: # Atom
        return random.choice(['True', 'False'])
    elif choice < 0.4: # Unary NOT
        return f"NOT {generate_expression(depth - 1)}"
    else: # Binary Op
        op = random.choice(OPS)
        left = generate_expression(depth - 1)
        right = generate_expression(depth - 1)
        return f"( {left} {op} {right} )"

def eval_bool_expr(expr):
    # Convert dataset format to Python syntax for evaluation
    # AND -> and
    # OR -> or
    # NOT -> not
    # XOR -> ^ (bitwise XOR works as logical XOR for booleans)
    py_expr = expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not").replace("XOR", "^")
    return eval(py_expr)

def generate_dataset(count, max_depth):
    samples = []
    seen = set()
    
    while len(samples) < count:
        # Random depth up to max_depth
        depth = random.randint(1, max_depth)
        expr = generate_expression(depth)
        
        try:
            res = eval_bool_expr(expr)
            
            # Canonical spacing is already handled by f-strings in generate_expression
            # but let's ensure it's clean
            
            line = f"{expr} = {res}"
            
            if line not in seen:
                seen.add(line)
                samples.append(line)
        except:
            continue
            
    return samples

def main():
    print(f"Generating {args.split} dataset with max_depth={args.max_depth}...")
    
    if args.ood_depth:
        # OOD Depth: depth in [D_train+1, D_train+K]
        # Assuming D_train is passed as max_depth, we go deeper
        min_d = args.max_depth + 1
        max_d = args.max_depth + 3
        samples = []
        while len(samples) < args.count:
            depth = random.randint(min_d, max_d)
            expr = generate_expression(depth)
            try:
                res = eval_bool_expr(expr)
                line = f"{expr} = {res}"
                samples.append(line)
            except:
                continue
    elif args.ood_length:
        # OOD Length: longer sequences
        # We can achieve this by forcing deeper trees or chaining
        # For simplicity, let's just use a much larger depth
        samples = []
        while len(samples) < args.count:
            depth = random.randint(args.max_depth + 2, args.max_depth + 5)
            expr = generate_expression(depth)
            try:
                res = eval_bool_expr(expr)
                line = f"{expr} = {res}"
                samples.append(line)
            except:
                continue
    else:
        samples = generate_dataset(args.count, args.max_depth)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for line in samples:
            f.write(line + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
