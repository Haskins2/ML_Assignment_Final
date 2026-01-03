#!/usr/bin/env python3

import argparse
from pathlib import Path
import random

# argparse practice
parser = argparse.ArgumentParser(description="Generate a simple boolean logic dataset.")
parser.add_argument("--basic", action="store_true", help="Generate basic boolean problems.")
parser.add_argument("--advanced", action="store_true", help="Generate advanced boolean problems.")
args = parser.parse_args()


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_NAME = "input_boolean.txt"
OUTPUT_FILE = DATASET_DIR / DATASET_NAME

def eval_bool_expr(expr):
    # Convert dataset format to Python syntax for evaluation
    # AND -> and
    # OR -> or
    # NOT -> not
    # XOR -> ^ (bitwise XOR works as logical XOR for booleans)
    # Note: Replace XOR first because it contains "OR"
    py_expr = expr.replace("XOR", "^").replace("AND", "and").replace("OR", "or").replace("NOT", "not")
    return eval(py_expr)

def generate_basic_boolean():
    samples = []
    booleans = [True, False]
    
    # NOT
    for a in booleans:
        expr = f"NOT {a}"
        res = eval_bool_expr(expr)
        samples.append(f"{expr}={res}")
        
    # AND, OR, XOR
    for op in ["AND", "OR", "XOR"]:
        for a in booleans:
            for b in booleans:
                expr = f"{a} {op} {b}"
                res = eval_bool_expr(expr)
                samples.append(f"{expr}={res}")
    return samples

def generate_complex_boolean(count=5000):
    samples = []
    ops = ['AND', 'OR', 'XOR']
    booleans = [True, False]
    
    for _ in range(count):
        a = random.choice(booleans)
        b = random.choice(booleans)
        c = random.choice(booleans)
        
        op1 = random.choice(ops)
        op2 = random.choice(ops)
        
        # Patterns:
        # 1. a op1 b op2 c
        # 2. (a op1 b) op2 c
        # 3. a op1 (b op2 c)
        # 4. NOT a op1 b
        # 5. a op1 NOT b
        
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
            samples.append(f"{expr}={res}")
        except:
            pass
            
    return samples

def jumble_samples(samples):
    random.shuffle(samples)
    return samples


def main():
    samples = []
    if args.basic:
        print("\nGenerating basic boolean problems...")
        # Generate many copies since the space is small
        basic_set = generate_basic_boolean()
        # Repeat to get a decent dataset size
        for _ in range(1000):
            samples.extend(basic_set)
            
    elif args.advanced:
        print("\nGenerating advanced boolean problems...")
        # Include basics
        basic_set = generate_basic_boolean()
        for _ in range(100):
             samples.extend(basic_set)
             
        samples += generate_complex_boolean(count=10000)
    else:
        print("\nSpecify either --basic or --advanced")
        return
    
    samples = jumble_samples(samples)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.unlink(missing_ok=True) # overwrite

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for line in samples:
            f.write(line + "\n")

    print(f"\nWrote {len(samples)} samples to datasets/{DATASET_NAME}\n")


if __name__ == "__main__":
    main()