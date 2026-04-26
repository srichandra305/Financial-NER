"""
utils/preprocessing.py
-----------------------
Text preprocessing and tokenisation utilities for the Financial NER pipeline.
"""

import re
import string
from typing import List, Tuple, Dict

# ── Constants ──────────────────────────────────────────────────────────────────
CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹", "₿", "¢"}
COMMON_TICKERS   = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA",
    "NFLX", "BABA", "JPM", "GS", "V", "MA", "BRK", "PFE", "JNJ",
    "XOM", "CVX", "WMT", "UNH", "BAC", "T", "VZ", "DIS", "PYPL",
    "INTC", "AMD", "CRM", "ORCL", "IBM", "QCOM", "TXN", "AVGO",
    "BTC", "ETH", "SOL", "DOGE", "BNB", "XRP",
}
MACRO_INDICATORS = {
    "GDP", "CPI", "PPI", "PMI", "NFP", "EPS", "PE", "ROE", "ROA",
    "EBITDA", "EBIT", "FCF", "DCF", "YOY", "QOQ", "MOM",
}
FINANCIAL_EVENTS = {
    "merger", "acquisition", "ipo", "dividend", "split", "bankruptcy",
    "delisting", "earnings", "buyback", "spinoff", "listing", "ipo",
    "bond", "offering", "recall", "settlement", "fine", "penalty",
}


# ── Tokeniser ──────────────────────────────────────────────────────────────────

def tokenise(text: str) -> List[str]:
    """
    Rule-based financial tokeniser.
    - Preserves currency symbols as separate tokens.
    - Keeps ticker-like ALL-CAPS words intact.
    - Splits on whitespace and common punctuation.
    """
    # Insert spaces around currency symbols and punctuation
    text = re.sub(r"([$€£¥₹₿¢])", r" \1 ", text)
    text = re.sub(r"([.,;:!?()\[\]{}'\"\\])", r" \1 ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return [t for t in tokens if t]


def clean_token(token: str) -> str:
    """Lowercase and strip leading/trailing punctuation from a token."""
    return token.strip(string.punctuation).lower()


# ── Feature builder (mirrors train.py logic) ───────────────────────────────────

def build_token_features(token: str, prev_tok: str = "<START>", next_tok: str = "<END>") -> str:
    """
    Build a feature string for a single token with context.
    Must stay in sync with models/train.py: build_token_features().
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
        feats.append("SHORT_UPPER")
    if any(c.isdigit() for c in token):
        feats.append("HAS_DIGIT")
    feat_str = (
        " ".join(feats)
        + " PREV:" + prev_tok.lower()
        + " NEXT:" + next_tok.lower()
    )
    return feat_str


def build_feature_sequence(tokens: List[str]) -> List[str]:
    """Build a feature string for every token in a sentence."""
    feats = []
    n = len(tokens)
    for i, tok in enumerate(tokens):
        prev_tok = tokens[i - 1] if i > 0 else "<START>"
        next_tok = tokens[i + 1] if i < n - 1 else "<END>"
        feats.append(build_token_features(tok, prev_tok, next_tok))
    return feats


# ── Rule-based pre-filters ─────────────────────────────────────────────────────

def rule_based_label(token: str) -> str | None:
    """
    Fast heuristic to assign a label without the ML model.
    Returns None if no rule fires (defer to model).
    """
    t = token.strip()
    if t in CURRENCY_SYMBOLS:
        return "CURRENCY"
    if t.upper() in COMMON_TICKERS:
        return "TICKER"
    if t.upper() in MACRO_INDICATORS:
        return "INDICATOR"
    if clean_token(t) in FINANCIAL_EVENTS:
        return "EVENT"
    return None


# ── Entity aggregation ─────────────────────────────────────────────────────────

def aggregate_entities(
    tokens: List[str], labels: List[str]
) -> List[Dict]:
    """
    Merge consecutive tokens that share the same non-O label into spans.

    Returns a list of dicts:
        {"text": str, "label": str, "start": int, "end": int}
    where start/end are token indices (inclusive).
    """
    entities = []
    i = 0
    while i < len(tokens):
        lbl = labels[i]
        if lbl != "O":
            j = i + 1
            while j < len(tokens) and labels[j] == lbl:
                j += 1
            span_text = " ".join(tokens[i:j])
            entities.append({
                "text":  span_text,
                "label": lbl,
                "start": i,
                "end":   j - 1,
            })
            i = j
        else:
            i += 1
    return entities


# ── Character-offset mapping ───────────────────────────────────────────────────

def tokens_to_char_offsets(
    text: str, tokens: List[str]
) -> List[Tuple[int, int]]:
    """
    Map each token back to its (start, end) character offset in the original text.
    Useful for front-end highlighting.
    """
    offsets = []
    cursor  = 0
    for tok in tokens:
        idx = text.find(tok, cursor)
        if idx == -1:
            # Fallback: try case-insensitive
            lo = text.lower().find(tok.lower(), cursor)
            idx = lo if lo != -1 else cursor
        offsets.append((idx, idx + len(tok)))
        cursor = idx + len(tok)
    return offsets
