# Boolean Logic GPT — Specification (Part 2)

## Objective

Train a small GPT-style transformer (based on `gpt.py`) to evaluate Boolean expressions.

Given an input expression, the model must output the correct Boolean value:
`<EXPR> = <BOOL>`

Examples:

- `True AND False = False`
- `NOT True = False`
- `( True OR False ) AND True = True`
- `True XOR True = False`

This is Part 2 of the CS7CS4/CSU44061 final assignment.

## Scope

- Literals: `True`, `False`
- Operators: `AND`, `OR`, `NOT`, `XOR`
- Parentheses are supported.
- Output is a single token: `True` or `False`.

## Data format

Each training sample is a token sequence:
`<expr_tokens> + ['='] + [label_token] + ['\n']`

We canonicalise formatting so every token is separated by a single space in the rendered text form.
Example tokenisation:
`( True OR False ) XOR False = False \n`

## Tokeniser

Use a token-level vocabulary:

Tokens:

- `True`, `False`
- `AND`, `OR`, `NOT`, `XOR`
- `(`, `)`, `=`, `\n`

Encoding: split on spaces for the canonicalised samples.
Decoding: join with spaces, with `\n` treated as end-of-sample.

No character-level modelling for Part 2.

## Dataset generation

Generate synthetic expressions from a grammar:

atom := True | False | ( expr ) | NOT atom
expr := atom | ( atom op atom ) | ( expr op expr )
op ∈ {AND, OR, XOR}

Generation controls:

- Max depth D (train: D_train, OOD-depth test: D_ood > D_train)
- Optional max token length L
- Operator frequency balancing
- Label balancing to keep ~50/50 True/False

Evaluation:

- Deterministic evaluation of the expression tree to obtain the label.

Splits:

- Train / Val / Test (IID): depth ≤ D_train
- Test-OOD-Depth: depth in [D_train+1, D_train+K]
- Test-OOD-Length: longer sequences than training

## Model

Base architecture: `GPTLanguageModel` from `gpt.py`.

Recommended starting hyperparameters:

- block_size: 64 (tune based on max expression length)
- n_layer: 4
- n_head: 4
- n_embd: 128
- dropout: 0.1–0.2
- batch_size: 64
- AdamW, LR ~3e-4 with optional warmup/cosine decay

## Training objective

Primary objective: predict the correct label token after `=`.

Two modes (experiment):

1. Standard next-token cross-entropy over the entire sequence.
2. Masked cross-entropy computed ONLY on the label position (token immediately after `=`),
   optionally also on the trailing `\n`.

We will compare both modes in accuracy and convergence speed.

## Evaluation metrics

Report:

- Label accuracy (%) on IID Test
- Label accuracy (%) on OOD-Depth and OOD-Length
- Error breakdown by:
  - operator set present (AND/OR/XOR/NOT)
  - depth bucket
  - token length bucket

Evaluation protocol:

- Provide the model with tokens up to `=`
- Generate exactly 1 token
- Compare to ground truth (`True` or `False`)

## Deliverables

- `model_weights_part2.pth` (final trained model weights)
- Code to generate datasets and train/evaluate end-to-end
- Appendix: a brief selection of prompt → output examples highlighting strengths/weaknesses
- Report section describing:
  - dataset design
  - architecture adaptations (tokeniser, block_size, loss masking, regularisation)
  - quantitative comparisons and qualitative failure cases

## Files / Modules (planned)

- `boolean_dataset.py`
  - expression tree generator
  - renderer (canonical token spacing)
  - evaluator
  - dataset writer for train/val/test + OOD splits
- `train_boolean.py`
  - loads/generates dataset
  - builds vocab and enc/dec
  - trains `GPTLanguageModel`
  - saves `model_weights_part2.pth`
- `evaluate_boolean.py`
  - label-accuracy evaluation for IID + OOD
  - prints example failures by category
