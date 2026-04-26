"""
backend/main.py
---------------
FastAPI application for the Financial NER System.
Endpoints:
  GET  /            → health check
  POST /predict     → NER prediction
  GET  /labels      → label metadata
  GET  /model/info  → loaded model details
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Make project root importable when running from backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.preprocessing import (
    tokenise,
    build_feature_sequence,
    aggregate_entities,
    tokens_to_char_offsets,
)
from utils.entity_utils import (
    LABEL_META,
    ALL_LABELS,
    format_entity_response,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
META_PATH  = PROJECT_ROOT / "models" / "model_meta.json"

# ── Load model at startup ──────────────────────────────────────────────────────
_pipeline   = None
_model_meta = {"best_model": "unknown", "weighted_f1": 0.0}


def load_model():
    global _pipeline, _model_meta
    if not MODEL_PATH.exists():
        log.warning(
            "Model file not found at %s. "
            "Run `python models/train.py` first.",
            MODEL_PATH,
        )
        return
    with open(MODEL_PATH, "rb") as fh:
        _pipeline = pickle.load(fh)
    log.info("Model loaded from %s", MODEL_PATH)

    if META_PATH.exists():
        with open(META_PATH) as fh:
            _model_meta = json.load(fh)
    log.info("Best model: %s  F1=%.4f", _model_meta.get("best_model"), _model_meta.get("weighted_f1", 0))


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial NER API",
    description="Extract COMPANY · TICKER · EVENT · CURRENCY · INDICATOR from financial text.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_model()


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="Apple Inc reported record earnings of $97 billion this quarter.",
    )
    filter_labels: Optional[List[str]] = Field(
        default=None,
        description="Optional list of labels to keep. E.g. ['COMPANY','TICKER']",
    )


class TokenDetail(BaseModel):
    token:      str
    label:      str
    color:      str
    bg:         str
    icon:       str
    char_start: int
    char_end:   int


class EntitySpan(BaseModel):
    text:        str
    label:       str
    color:       str
    bg:          str
    icon:        str
    token_start: int
    token_end:   int


class PredictResponse(BaseModel):
    input_text: str
    tokens:     List[TokenDetail]
    entities:   List[EntitySpan]
    summary:    dict
    model_used: str
    label_meta: dict


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {
        "status":       "ok",
        "model_loaded": _pipeline is not None,
        "model_name":   _model_meta.get("best_model", "none"),
    }


@app.get("/labels", tags=["Metadata"])
def get_labels():
    """Return label set with colours, icons and descriptions."""
    return {k: v for k, v in LABEL_META.items() if k != "O"}


@app.get("/model/info", tags=["Metadata"])
def model_info():
    """Return info about the currently loaded model."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    return _model_meta


@app.post("/predict", response_model=PredictResponse, tags=["NER"])
def predict(req: PredictRequest):
    """
    Extract named entities from financial text.

    - **text**: Input sentence(s) to analyse (max 5 000 chars).
    - **filter_labels**: Optional list of entity labels to return.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python models/train.py` then restart the server.",
        )

    text   = req.text.strip()
    tokens = tokenise(text)

    if not tokens:
        raise HTTPException(status_code=400, detail="No tokens extracted from input.")

    # Build feature strings and predict
    feature_strings = build_feature_sequence(tokens)
    try:
        labels: List[str] = _pipeline.predict(feature_strings).tolist()
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Aggregate consecutive same-label spans
    entities = aggregate_entities(tokens, labels)

    # Optional label filter
    if req.filter_labels:
        allowed = set(req.filter_labels)
        entities = [e for e in entities if e["label"] in allowed]
        labels   = [lbl if lbl in allowed else "O" for lbl in labels]

    # Map tokens back to character offsets
    offsets = tokens_to_char_offsets(text, tokens)

    response = format_entity_response(
        tokens=tokens,
        labels=labels,
        entities=entities,
        text=text,
        offsets=offsets,
        model_name=_model_meta.get("best_model", "unknown"),
    )

    return JSONResponse(content=response)


# ── Dev entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
