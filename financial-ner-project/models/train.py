"""
Financial NER Training Script
Trains multiple classifiers for token-level NER using TF-IDF features.
Best model is saved to models/best_model.pkl
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "sample_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "models", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Label set ──────────────────────────────────────────────────────────────────
LABELS = ["O", "COMPANY", "TICKER", "EVENT", "CURRENCY", "INDICATOR"]

# ── Feature engineering ────────────────────────────────────────────────────────

def build_token_features(token: str) -> str:
    """
    Enrich a token with hand-crafted orthographic / positional signals
    returned as a single string that TF-IDF can digest.
    """
    feats = [token.lower()]
    if token.isupper():
        feats.append("ALL_UPPER")
    if token.istitle():
        feats.append("IS_TITLE")
    if token.isdigit():
        feats.append("IS_DIGIT")
    if token.startswith("$"):
        feats.append("STARTS_DOLLAR")
    if token.startswith("%"):
        feats.append("ENDS_PERCENT")
    if len(token) <= 5 and token.isupper():
        feats.append("SHORT_UPPER")   # likely ticker
    if any(c.isdigit() for c in token):
        feats.append("HAS_DIGIT")
    return " ".join(feats)


def load_and_prepare(path: str):
    """
    Load the BIO-tagged CSV, produce (X_text, y_label) arrays.

    Expected CSV columns:
        sentence_id | token | label
    """
    df = pd.read_csv(path)
    df.dropna(subset=["token", "label"], inplace=True)

    # Group by sentence to build context window features
    X, y = [], []
    for sid, grp in df.groupby("sentence_id"):
        tokens = grp["token"].tolist()
        labels = grp["label"].tolist()
        for i, (tok, lbl) in enumerate(zip(tokens, labels)):
            prev_tok = tokens[i - 1] if i > 0 else "<START>"
            next_tok = tokens[i + 1] if i < len(tokens) - 1 else "<END>"
            feat_str = (
                build_token_features(tok)
                + " PREV:" + prev_tok.lower()
                + " NEXT:" + next_tok.lower()
            )
            X.append(feat_str)
            y.append(lbl)

    return np.array(X), np.array(y)


# ── Models ─────────────────────────────────────────────────────────────────────

def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1
        ),
        "RidgeClassifier": RidgeClassifier(alpha=1.0),
        "LinearSVC": LinearSVC(max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
        ),
    }


def build_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=50_000,
            sublinear_tf=True,
        )),
        ("clf", clf),
    ])


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test, name: str):
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.4f}   Weighted-F1 : {f1:.4f}")
    print(f"{'='*60}")
    print(report)

    # Save text report
    rpt_path = os.path.join(REPORT_DIR, f"{name}_report.txt")
    with open(rpt_path, "w") as fh:
        fh.write(f"{name}\n{'='*60}\n")
        fh.write(f"Accuracy : {acc:.4f}   Weighted-F1 : {f1:.4f}\n\n")
        fh.write(report)

    # Confusion matrix
    labels_present = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_present)
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    cm_path = os.path.join(REPORT_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close()

    return f1, acc


def plot_comparison(results: dict):
    names  = list(results.keys())
    f1s    = [v["f1"]  for v in results.values()]
    accs   = [v["acc"] for v in results.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, f1s,  width, label="Weighted F1", color="#3b82f6")
    bars2 = ax.bar(x + width / 2, accs, width, label="Accuracy",    color="#10b981")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – Financial NER")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3)
    ax.bar_label(bars2, fmt="%.3f", padding=3)
    plt.tight_layout()
    cmp_path = os.path.join(REPORT_DIR, "model_comparison.png")
    plt.savefig(cmp_path, dpi=120)
    plt.close()
    print(f"\nComparison chart saved → {cmp_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    X, y = load_and_prepare(DATA_PATH)
    print(f"  Total tokens : {len(X)}")
    print(f"  Label dist   : {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    classifiers = get_classifiers()
    results     = {}
    best_f1     = -1.0
    best_name   = None
    best_pipe   = None

    for name, clf in classifiers.items():
        print(f"\nTraining {name} …")
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)

        f1, acc = evaluate(pipe, X_test, y_test, name)
        results[name] = {"f1": f1, "acc": acc}

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_pipe = pipe

    plot_comparison(results)

    # ── Save best model ────────────────────────────────────────────────────────
    model_path   = os.path.join(MODEL_DIR, "best_model.pkl")
    meta_path    = os.path.join(MODEL_DIR, "model_meta.json")

    with open(model_path, "wb") as fh:
        pickle.dump(best_pipe, fh)

    with open(meta_path, "w") as fh:
        json.dump({
            "best_model": best_name,
            "weighted_f1": round(best_f1, 4),
            "results": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                        for k, v in results.items()},
        }, fh, indent=2)

    print(f"\n✓ Best model : {best_name}  (F1={best_f1:.4f})")
    print(f"  Saved → {model_path}")
    print(f"  Meta  → {meta_path}")


if __name__ == "__main__":
    main()
