#!/usr/bin/env python3

import argparse
from pathlib import Path

# argparse practice
parser = argparse.ArgumentParser(description="Generate a simple math dataset.")
parser.add_argument("--basic", action="store_true", help="Generate basic single-digit addition problems.")
parser.add_argument("--advanced", action="store_true", help="Generate advanced math problems (TODO).")
args = parser.parse_args()


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_NAME = "input_math.txt"
OUTPUT_FILE = DATASET_DIR / DATASET_NAME

MIN_DIGIT = 0
MAX_DIGIT = 9

def generate_single_digit_addition():
    samples = []

    for a in range(MIN_DIGIT, MAX_DIGIT + 1):
        for b in range(MIN_DIGIT, MAX_DIGIT + 1):
            result = a + b
            samples.append(f"{a}+{b}={result}")

    return samples

def generate_single_digit_subtraction():
    samples = []

    for a in range(MIN_DIGIT, MAX_DIGIT + 1):
        for b in range(MIN_DIGIT, MAX_DIGIT + 1):
            result = a - b
            samples.append(f"{a}-{b}={result}")

    return samples

def generate_single_digit_multiplication():
    samples = []

    for a in range(MIN_DIGIT, MAX_DIGIT + 1):
        for b in range(MIN_DIGIT, MAX_DIGIT + 1):
            result = a * b
            samples.append(f"{a}*{b}={result}")

    return samples

def generate_single_digit_division():
    samples = []

    for a in range(MIN_DIGIT, MAX_DIGIT + 1):
        for b in range(1, MAX_DIGIT + 1):  # avoid division by zero
            result = a // b
            samples.append(f"{a}/{b}={result}")

    return samples

def jumble_samples(samples):
    import random
    random.shuffle(samples)
    return samples


def main():
    if args.basic:
        print("\nGenerating basic single-digit addition problems...")
        samples = generate_single_digit_addition()
        samples += generate_single_digit_subtraction()
        samples += generate_single_digit_multiplication()
        samples += generate_single_digit_division()
    elif args.advanced:
        print("\nAdvanced math problem generation is not yet implemented.")
        return
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