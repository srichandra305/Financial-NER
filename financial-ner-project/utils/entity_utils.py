"""
utils/entity_utils.py
---------------------
Entity colour mapping, label metadata, and response formatting helpers.
"""

from typing import List, Dict, Any

# ── Label metadata ─────────────────────────────────────────────────────────────

LABEL_META: Dict[str, Dict[str, str]] = {
    "COMPANY": {
        "color":       "#3b82f6",   # blue
        "bg":          "#eff6ff",
        "description": "Organisation / corporation name",
        "icon":        "🏢",
    },
    "TICKER": {
        "color":       "#8b5cf6",   # purple
        "bg":          "#f5f3ff",
        "description": "Stock / crypto exchange symbol",
        "icon":        "📈",
    },
    "EVENT": {
        "color":       "#f59e0b",   # amber
        "bg":          "#fffbeb",
        "description": "Corporate or economic event",
        "icon":        "⚡",
    },
    "CURRENCY": {
        "color":       "#10b981",   # emerald
        "bg":          "#ecfdf5",
        "description": "Currency symbol, amount, or cryptocurrency",
        "icon":        "💵",
    },
    "INDICATOR": {
        "color":       "#ef4444",   # red
        "bg":          "#fef2f2",
        "description": "Macroeconomic or market indicator",
        "icon":        "📊",
    },
    "O": {
        "color":       "#6b7280",
        "bg":          "transparent",
        "description": "Non-entity token",
        "icon":        "",
    },
}

ALL_LABELS = [k for k in LABEL_META if k != "O"]


def label_color(label: str) -> str:
    return LABEL_META.get(label, LABEL_META["O"])["color"]


def label_bg(label: str) -> str:
    return LABEL_META.get(label, LABEL_META["O"])["bg"]


def label_icon(label: str) -> str:
    return LABEL_META.get(label, LABEL_META["O"])["icon"]


# ── Response formatting ────────────────────────────────────────────────────────

def format_entity_response(
    tokens: List[str],
    labels: List[str],
    entities: List[Dict],
    text: str,
    offsets: List[tuple],
    model_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Build the full JSON response consumed by the frontend.
    """
    # Per-token detail
    token_details = [
        {
            "token":  tok,
            "label":  lbl,
            "color":  label_color(lbl),
            "bg":     label_bg(lbl),
            "icon":   label_icon(lbl),
            "char_start": offsets[i][0] if i < len(offsets) else 0,
            "char_end":   offsets[i][1] if i < len(offsets) else 0,
        }
        for i, (tok, lbl) in enumerate(zip(tokens, labels))
    ]

    # Entity spans
    entity_spans = [
        {
            "text":        ent["text"],
            "label":       ent["label"],
            "color":       label_color(ent["label"]),
            "bg":          label_bg(ent["label"]),
            "icon":        label_icon(ent["label"]),
            "token_start": ent["start"],
            "token_end":   ent["end"],
        }
        for ent in entities
        if ent["label"] != "O"
    ]

    # Summary counts
    summary: Dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}
    for ent in entity_spans:
        summary[ent["label"]] = summary.get(ent["label"], 0) + 1

    return {
        "input_text":    text,
        "tokens":        token_details,
        "entities":      entity_spans,
        "summary":       summary,
        "model_used":    model_name,
        "label_meta":    {k: v for k, v in LABEL_META.items() if k != "O"},
    }


def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entity spans (same text + label)."""
    seen = set()
    unique = []
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    return unique


def filter_by_label(entities: List[Dict], labels: List[str]) -> List[Dict]:
    """Keep only entities whose label is in *labels*."""
    label_set = set(labels)
    return [e for e in entities if e["label"] in label_set]
