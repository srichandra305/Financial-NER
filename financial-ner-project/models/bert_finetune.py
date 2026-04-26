"""
models/bert_finetune.py
-----------------------
Optional HuggingFace DistilBERT fine-tuning for Financial NER.

Usage:
    pip install transformers torch datasets seqeval
    python models/bert_finetune.py

The script:
1. Loads the same CSV dataset used by train.py
2. Converts it to a HuggingFace Dataset with BIO tags
3. Fine-tunes DistilBERT-base-uncased for token classification
4. Evaluates using seqeval
5. Saves the fine-tuned model to models/bert_ner/

Note: GPU recommended. CPU training will be slow (~30 min for 3 epochs).
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sample_dataset.csv"
OUT_DIR   = BASE_DIR / "models" / "bert_ner"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Label map ──────────────────────────────────────────────────────────────────
RAW_LABELS = ["O", "COMPANY", "TICKER", "EVENT", "CURRENCY", "INDICATOR"]

# Build BIO label list
BIO_LABELS: List[str] = ["O"]
for lbl in RAW_LABELS[1:]:
    BIO_LABELS.append(f"B-{lbl}")
    BIO_LABELS.append(f"I-{lbl}")

LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

print(f"BIO label set ({len(BIO_LABELS)}): {BIO_LABELS}")


# ── Build BIO-tagged sentences ─────────────────────────────────────────────────

def raw_to_bio(tokens: List[str], labels: List[str]) -> List[str]:
    """Convert flat labels to BIO scheme."""
    bio = []
    prev = "O"
    for tok, lbl in zip(tokens, labels):
        if lbl == "O":
            bio.append("O")
            prev = "O"
        elif lbl != prev:
            bio.append(f"B-{lbl}")
            prev = lbl
        else:
            bio.append(f"I-{lbl}")
    return bio


def load_sentences(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    df = pd.read_csv(path)
    df.dropna(subset=["token", "label"], inplace=True)

    all_tokens, all_labels = [], []
    for _, grp in df.groupby("sentence_id"):
        tokens = grp["token"].tolist()
        labels = grp["label"].tolist()
        bio    = raw_to_bio(tokens, labels)
        all_tokens.append(tokens)
        all_labels.append(bio)
    return all_tokens, all_labels


# ── Main fine-tuning routine ───────────────────────────────────────────────────

def main():
    # Lazy-import heavy deps so the file can be imported without them installed
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
        )
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        print(f"[bert_finetune] Missing dependency: {exc}")
        print("Install with: pip install transformers torch datasets seqeval")
        sys.exit(1)

    MODEL_NAME = "distilbert-base-uncased"
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    print("Loading dataset …")
    all_tokens, all_labels = load_sentences(DATA_PATH)
    print(f"  {len(all_tokens)} sentences loaded")

    # Train / eval split (80/20)
    split = int(len(all_tokens) * 0.8)
    train_tokens, eval_tokens = all_tokens[:split], all_tokens[split:]
    train_labels, eval_labels = all_labels[:split], all_labels[split:]

    # Tokenise and align labels
    def tokenize_and_align(token_lists, label_lists):
        encoding = tokenizer(
            token_lists,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        aligned_labels = []
        for i, labels in enumerate(label_lists):
            word_ids = encoding.word_ids(batch_index=i)
            prev_wid = None
            row = []
            for wid in word_ids:
                if wid is None:
                    row.append(-100)
                elif wid != prev_wid:
                    row.append(LABEL2ID[labels[wid]])
                else:
                    # continuation sub-token: use I- label if B- was assigned
                    lbl = labels[wid]
                    if lbl != "O":
                        i_lbl = f"I-{lbl[2:]}" if lbl.startswith("B-") else lbl
                        row.append(LABEL2ID.get(i_lbl, LABEL2ID[lbl]))
                    else:
                        row.append(-100)
                prev_wid = wid
            aligned_labels.append(row)
        encoding["labels"] = aligned_labels
        return encoding

    print("Tokenising …")
    train_enc = tokenize_and_align(train_tokens, train_labels)
    eval_enc  = tokenize_and_align(eval_tokens,  eval_labels)

    train_dataset = Dataset.from_dict(train_enc)
    eval_dataset  = Dataset.from_dict(eval_enc)
    dataset       = DatasetDict({"train": train_dataset, "validation": eval_dataset})

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Compute metrics
    try:
        from seqeval.metrics import f1_score as seq_f1, classification_report as seq_report
        HAS_SEQEVAL = True
    except ImportError:
        HAS_SEQEVAL = False
        print("seqeval not installed – metrics will be skipped (pip install seqeval)")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_preds, true_labels = [], []
        for pred_row, label_row in zip(predictions, labels):
            tp, tl = [], []
            for pred, label in zip(pred_row, label_row):
                if label != -100:
                    tp.append(ID2LABEL[pred])
                    tl.append(ID2LABEL[label])
            true_preds.append(tp)
            true_labels.append(tl)
        if HAS_SEQEVAL:
            return {"f1": seq_f1(true_labels, true_preds)}
        return {}

    # Training args
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if HAS_SEQEVAL else "eval_loss",
        logging_dir=str(OUT_DIR / "logs"),
        logging_steps=10,
        report_to="none",
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if HAS_SEQEVAL else None,
    )

    print("\nStarting fine-tuning …")
    trainer.train()

    print("\nEvaluating …")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    # Save
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\n✓ BERT model saved → {OUT_DIR}")

    # Save label map
    with open(OUT_DIR / "label_map.json", "w") as fh:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, fh, indent=2)
    print(f"  Label map saved → {OUT_DIR / 'label_map.json'}")


if __name__ == "__main__":
    main()
