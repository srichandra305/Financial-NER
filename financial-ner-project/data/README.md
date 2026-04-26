# Dataset Format

## File: `sample_dataset.csv`

### Columns

| Column        | Type    | Description                                      |
|---------------|---------|--------------------------------------------------|
| `sentence_id` | integer | Groups tokens that belong to the same sentence   |
| `token`       | string  | A single whitespace-tokenised word/punctuation   |
| `label`       | string  | BIO-lite entity label (no B-/I- prefix needed)   |

### Label Set

| Label       | Meaning                                                        | Example tokens              |
|-------------|----------------------------------------------------------------|-----------------------------|
| `O`         | Outside – not an entity                                        | "the", "reported", "."      |
| `COMPANY`   | Organisation name (public or private)                          | Apple, Goldman, Sachs       |
| `TICKER`    | Exchange ticker symbol                                         | AAPL, TSLA, NVDA            |
| `EVENT`     | Corporate/economic event                                       | merger, IPO, dividend       |
| `CURRENCY`  | Currency symbol, amount, or crypto                             | $, 97, billion, EUR, BTC    |
| `INDICATOR` | Macroeconomic or market indicator                              | GDP, CPI, S&P 500, yield    |

### Notes
- Each row is **one token**.
- Sentences are grouped by `sentence_id` – the model uses a ±1 context window.
- Multi-token entities are labelled on each token (flat labelling, no BIO prefix).
- Extend the dataset by appending more rows; keep `sentence_id` monotonically increasing.
