#!/usr/bin/env python3

from pathlib import Path

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


def main():
    samples = generate_single_digit_addition()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.unlink(missing_ok=True) # overwrite

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for line in samples:
            f.write(line + "\n")

    print(f"\nWrote {len(samples)} samples to datasets/{DATASET_NAME}\n")


if __name__ == "__main__":
    main()