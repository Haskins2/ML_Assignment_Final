#!/usr/bin/env python3

import argparse
from pathlib import Path
import random

# argparse practice
parser = argparse.ArgumentParser(description="Generate a simple math dataset.")
parser.add_argument("--basic", action="store_true", help="Generate basic single-digit math problems.")
parser.add_argument("--advanced", action="store_true", help="Generate advanced two-digit math problems.")
args = parser.parse_args()


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_NAME = "input_math.txt"
OUTPUT_FILE = DATASET_DIR / DATASET_NAME

def generate_addition(min_val, max_val):
    samples = []
    for a in range(min_val, max_val + 1):
        for b in range(min_val, max_val + 1):
            result = a + b
            samples.append(f"{a}+{b}={result}")
    return samples

def generate_subtraction(min_val, max_val):
    samples = []
    for a in range(min_val, max_val + 1):
        for b in range(min_val, max_val + 1):
            result = a - b
            samples.append(f"{a}-{b}={result}")
    return samples

def generate_multiplication(min_val, max_val):
    samples = []
    for a in range(min_val, max_val + 1):
        for b in range(min_val, max_val + 1):
            result = a * b
            samples.append(f"{a}*{b}={result}")
    return samples

def generate_division(min_val, max_val):
    samples = []
    for a in range(min_val, max_val + 1):
        for b in range(1, max_val + 1):  # avoid division by zero
            if a % b == 0:
                result = a // b
                samples.append(f"{a}/{b}={result}")
    return samples

def jumble_samples(samples):
    random.shuffle(samples)
    return samples


def main():
    samples = []
    if args.basic:
        print("\nGenerating basic single-digit math problems...")
        min_val, max_val = 0, 9
        samples += generate_addition(min_val, max_val)
        samples += generate_subtraction(min_val, max_val)
        samples += generate_multiplication(min_val, max_val)
        samples += generate_division(min_val, max_val)
    elif args.advanced:
        print("\nGenerating advanced two-digit math problems...")
        min_val, max_val = 0, 99
        samples += generate_addition(min_val, max_val)
        samples += generate_subtraction(min_val, max_val)
        samples += generate_multiplication(min_val, max_val)
        samples += generate_division(min_val, max_val)
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