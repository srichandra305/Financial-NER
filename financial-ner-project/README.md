# 🔍 Financial NER — Named Entity Recognition System

Extract **COMPANY · TICKER · EVENT · CURRENCY · INDICATOR** entities from financial news text.

---

## 📁 Project Structure

```
financial-ner-project/
├── backend/
│   └── main.py               # FastAPI app (predict, health, metadata routes)
├── frontend/
│   └── index.html            # Single-file UI with entity highlighting
├── models/
│   ├── train.py              # Multi-model training script
│   ├── bert_finetune.py      # Optional HuggingFace DistilBERT fine-tuning
│   ├── best_model.pkl        # ← generated after training
│   ├── model_meta.json       # ← generated after training
│   └── reports/              # ← confusion matrices + comparison charts
├── data/
│   ├── sample_dataset.csv    # BIO-tagged training data
│   └── README.md             # Dataset format documentation
├── notebooks/
│   └── eda_and_evaluation.ipynb
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py      # Tokeniser, feature builder, span aggregator
│   └── entity_utils.py       # Label metadata, colour map, response formatter
├── requirements.txt
├── run.sh                    # Convenience launcher
└── README.md
```

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
# or
./run.sh install
```

### 2 — Train models

```bash
python models/train.py
# or
./run.sh train
```

This trains **Logistic Regression, Ridge Classifier, Linear SVC, Random Forest** and saves the best model (by weighted F1) to `models/best_model.pkl`. Evaluation charts are written to `models/reports/`.

### 3 — Start the API

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# or
./run.sh serve
```

API docs → [http://localhost:8000/docs](http://localhost:8000/docs)

### 4 — Open the frontend

Open `frontend/index.html` in any browser. No build step required.

```bash
./run.sh frontend
```

---

## 🧠 API Reference

### `POST /predict`

```json
// Request
{
  "text": "Apple Inc (AAPL) reported record earnings of $97 billion.",
  "filter_labels": null
}

// Response
{
  "input_text": "...",
  "tokens": [{ "token": "Apple", "label": "COMPANY", "color": "#3b82f6", ... }],
  "entities": [{ "text": "Apple Inc", "label": "COMPANY", ... }],
  "summary": { "COMPANY": 1, "TICKER": 1, "EVENT": 0, "CURRENCY": 3, "INDICATOR": 0 },
  "model_used": "LogisticRegression",
  "label_meta": { ... }
}
```

### `GET /labels`
Returns label colour map and descriptions.

### `GET /model/info`
Returns the currently loaded model name and its evaluation metrics.

### `GET /`
Health check.

---

## 🏷️ Entity Labels

| Label | Description | Example |
|-------|-------------|---------|
| `COMPANY` | Organisation name | Apple, Goldman Sachs |
| `TICKER` | Stock/crypto symbol | AAPL, BTC |
| `EVENT` | Corporate/economic event | merger, IPO, dividend |
| `CURRENCY` | Currency symbol or amount | $, 97 billion, EUR |
| `INDICATOR` | Macro/market indicator | GDP, CPI, S&P 500 |

---

## 🤖 Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Fast baseline, strong on high-dim sparse features |
| Ridge Classifier | Very fast linear model, good regularisation |
| Linear SVC | Often best on text classification tasks |
| Random Forest | Non-linear, captures feature interactions |
| DistilBERT (optional) | Best accuracy; requires GPU, ~30 min to fine-tune |

Features: character n-gram TF-IDF (2–5 grams) + orthographic signals + ±1 token context window.

---

## 📊 BERT Fine-tuning (optional)

```bash
pip install transformers torch datasets seqeval accelerate
python models/bert_finetune.py
# or
./run.sh bert
```

Fine-tunes `distilbert-base-uncased` for 3 epochs; model saved to `models/bert_ner/`.

---

## 🗂️ Dataset Format

See [`data/README.md`](data/README.md) for full schema.

CSV columns: `sentence_id | token | label`

Extend by appending rows; keep `sentence_id` monotonically increasing.

---

## 📓 Notebook

```bash
cd notebooks
jupyter notebook eda_and_evaluation.ipynb
```

Covers EDA, label distribution, feature inspection, live predictions, error analysis.

---

## ⚙️ Environment

- Python 3.10+
- CPU training: ~30 seconds for ML models
- GPU optional (only for BERT fine-tuning)

---

## 📄 License

MIT
