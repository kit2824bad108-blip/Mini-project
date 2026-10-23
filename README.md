# 🌾 Machine Learning Based Crop Yield Prediction Portal

An end-to-end, open-source crop yield prediction portal: a dependency-free web
frontend backed by a trained machine-learning service.

[![CI](https://github.com/kit2824bad108-blip/crop-yield-prediction-portal/actions/workflows/ci.yml/badge.svg)](https://github.com/kit2824bad108-blip/crop-yield-prediction-portal/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> Predict crop yield using advanced machine learning — smarter farming for a food-secure future.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [The interface](#the-interface)
6. [Training the model](#training-the-model)
7. [API reference](#api-reference)
8. [Testing](#testing)
9. [Docker](#docker)
10. [Deployment](#deployment)
11. [Design notes](#design-notes)
12. [Roadmap](#roadmap)
13. [License](#license)

---

## What it does

The portal turns agronomic conditions into a yield forecast:

| Input | Unit | Example |
|-------|------|---------|
| Rainfall | mm | 950 |
| Average temperature | °C | 24 |
| Soil N-P-K index | 0–100 | 72 |
| Irrigation | hours | 30 |
| Farm size | hectares | 50 |
| Crop | rice / wheat / maize / soybean | wheat |
| Season | rabi / kharif / zaid | kharif |

It returns a predicted yield in **tonnes per hectare**, the **estimated total
tonnage** for the given farm size, and which engine produced the number.

Two independent engines are available, and the UI always labels the one it used:

- **ML model** — the trained estimator served by the API (`model_used` names it).
- **Agronomic fallback** — a transparent response-factor model that guarantees
  the portal keeps working before training or when the API is unreachable.

---

## Architecture

```
                    ┌────────────────────────────────────┐
   Browser  ──────► │  index.html + styles.css + app.js  │
                    └───────────────┬────────────────────┘
                                    │ POST /predict (JSON)
                    ┌───────────────▼────────────────────┐
                    │  app.py — FastAPI + Uvicorn        │
                    │   /predict  /health  /models       │
                    │   also serves the static frontend  │
                    └───────────────┬────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  features.py — shared feature engineering │
              │  (imported by training *and* serving)     │
              └─────────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────▼─────────────────────┐
              │  models/best_xgboost.pkl                  │
              │  models/feature_columns.json              │
              │  models/metrics.json                      │
              └───────────────────────────────────────────┘
                          ▲ produced by
              ┌───────────┴───────────────────────────────┐
              │  train_model.py — trains & evaluates      │
              │  Random Forest · XGBoost · SVR · MLP      │
              └───────────────────────────────────────────┘
```

`features.py` is deliberately shared by both sides. Feature logic duplicated
between training and serving is the most common cause of silent ML regressions,
so this project has exactly one implementation.

---

## Project structure

```
.
├── index.html                     # Frontend markup
├── styles.css                     # Frontend styling (CSS custom properties)
├── app.js                         # Frontend logic (calls the API, falls back offline)
├── app.py                         # FastAPI service: inference API + static frontend
├── train_model.py                 # Training, evaluation and artifact saving
├── features.py                    # Shared feature engineering and reference model
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Runtime + test dependencies
├── pyproject.toml                 # pytest configuration
├── Dockerfile                     # Container image
├── tests/
│   ├── test_features.py           # Unit tests for the feature module
│   ├── test_api.py                # API + static-serving integration tests
│   └── test_trained_model.py      # Extra tests, active once a model is trained
├── .github/workflows/ci.yml       # CI: test matrix + train-and-test job
├── models/                        # Generated artifacts (git-ignored)
├── CONTRIBUTING.md
└── LICENSE
```

---

## Quick start

**Prerequisites:** Python 3.10 or newer.

```bash
git clone https://github.com/kit2824bad108-blip/crop-yield-prediction-portal.git
cd crop-yield-prediction-portal

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt

python train_model.py              # ~30 s: trains and saves the best model
python app.py                      # serves frontend + API on http://127.0.0.1:8000
```

Then open <http://127.0.0.1:8000>. The interactive API docs are at
<http://127.0.0.1:8000/docs>.

> The portal also runs **without** training: just `python app.py` and the
> fallback engine answers, with the UI labelling the result as an offline estimate.

---

## The interface

The portal is organised into four tabs, each driven by the same dependency-free
JavaScript:

| Tab | Contents |
|-----|----------|
| **Overview** | Hero artwork, the five-stage data-flow diagram, the pipeline steps and the benefit cards |
| **Performance** | Animated R² bar chart plus the full metrics table |
| **Predict Yield** | The prediction form with sliders and the live result panel |
| **Tech Stack** | How the pieces fit together, and the roadmap |

Fill in the conditions and press **Predict yield**. The result panel shows the
predicted yield in tonnes/hectare, a badge naming the engine that answered, and
the estimated total tonnage for the farm size you entered:

```
xgboost · 97% conf.        ≈ 69.2 tonnes across 50 ha
```

`app.js` posts to `/predict` on the same origin. If that request fails — the page
was opened from the filesystem, or hosted as a static site with no backend — it
recomputes the estimate with the same response-factor model the trainer uses and
labels the result `offline estimate · API unavailable`. It never silently
pretends an ML prediction happened.

### Accessibility and motion

- Tabs follow the [WAI-ARIA tabs pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/):
  `role="tablist"`, `aria-selected`, roving `tabindex`, and `ArrowLeft` /
  `ArrowRight` / `Home` / `End` keyboard navigation.
- The active tab is deep-linkable (`/#panel-predict`) and remembered in
  `localStorage`, so a reload returns you to where you were.
- Charts are labelled with `role="img"` and their numbers are duplicated in a
  real `<table>`, so the data is never trapped inside an image.
- All motion is opt-in: it runs only after `document.documentElement` receives a
  `js` class, and the whole animation layer is disabled under
  `prefers-reduced-motion: reduce`. Numbers, bars and panels still render at their
  final values — they simply stop animating into place.

### Static-only hosting

The frontend is three files with zero build step, so it works on GitHub Pages,
Netlify or Vercel as-is. In that mode every result is labelled an offline
estimate because there is no backend to call.

---

## Training the model

```bash
python train_model.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-samples` | `10000` | Number of synthetic farm records |
| `--models` | all four | Comma-separated subset of `random_forest,xgboost,svr,dnn` |
| `--output` | `models/best_xgboost.pkl` | Where to write the best model |
| `--seed` | `42` | Random seed |

The script fits every requested model, scores them on a 20% held-out split, and
saves the highest-R² model together with its feature schema and metrics:

```
models/best_xgboost.pkl      # the fitted estimator
models/feature_columns.json  # ordered schema the API must match
models/metrics.json          # per-model R2 / MAE / RMSE
```

Example run (`--n-samples 1500 --models xgboost`):

```
>>> Training xgboost ...
    R2: 0.9268 | MAE: 0.0501 | RMSE: 0.0628

Best model: xgboost (R2 = 0.9268)
[OK] Model saved          -> models/best_xgboost.pkl
[OK] Feature schema saved -> models/feature_columns.json
[OK] Metrics saved        -> models/metrics.json
```

All four models on a full 4,000-sample run:

| Model | R² | MAE | RMSE |
|-------|-----|------|------|
| Random Forest | 0.913 | 0.053 | 0.067 |
| **XGBoost** | **0.938** | **0.045** | **0.056** |
| Support Vector Regression | 0.779 | 0.085 | 0.106 |
| Neural Network (MLP) | 0.751 | 0.082 | 0.113 |

### Using your own data

`generate_synthetic_data()` exists so the repository is runnable with no
downloads. To train on real records, replace that function with a CSV read and
keep the column names used by `features.py`:

```python
def generate_synthetic_data(n_samples, seed=42):
    df = pd.read_csv("data/farm_records.csv")
    df = df.rename(columns={"rain_mm": "rainfall", "temp_c": "temperature"})
    df["crop"] = df["crop"].str.lower()
    df["season"] = df["season"].str.lower()
    return df
```

`yield_tph` must be the target column. Anything else is built by
`build_feature_frame()`.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | The portal frontend |
| `GET` | `/styles.css`, `/app.js` | Frontend assets |
| `GET` | `/health` | Readiness, active engine, `best_r2`, feature count |
| `GET` | `/api` | Service metadata |
| `GET` | `/models` | Discovered artifacts and per-model metrics |
| `POST` | `/predict` | Yield prediction |
| `GET` | `/docs` | Swagger UI |

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rainfall": 950,
    "temperature": 24,
    "soil_npk": 72,
    "irrigation_hours": 30,
    "farm_size_hectares": 50,
    "crop": "wheat",
    "season": "kharif"
  }'
```

```json
{
  "crop": "wheat",
  "season": "kharif",
  "predicted_yield_tph": 1.4,
  "estimated_total_tonnes": 69.83,
  "model_used": "xgboost",
  "confidence": 0.968,
  "inputs": {
    "rainfall": 950,
    "temperature": 24,
    "soil_npk": 72,
    "irrigation_hours": 30,
    "farm_size_hectares": 50,
    "crop": "wheat",
    "season": "kharif"
  },
  "timestamp": "2026-10-15T11:24:03.117Z"
}
```

`confidence` is derived from the selected model's R²; treat it as a rough
quality indicator, not a calibrated probability.

Requests are validated by Pydantic. Out-of-range numerics and unknown crops or
seasons return **422** with a descriptive body:

```json
{ "detail": [{ "loc": ["body", "crop"], "msg": "Input should be 'rice', 'wheat', 'maize' or 'soybean'", "type": "literal_error" }] }
```

---

## Testing

```bash
pytest
```

The suite is designed to run at any stage of the workflow:

- `tests/test_features.py` — feature encoding, validation, and the monotonic
  behaviour of the reference model.
- `tests/test_api.py` — endpoint contracts, input validation (422s), static-file
  serving, and confirmation that source files are *not* reachable over HTTP.
- `tests/test_trained_model.py` — automatically skipped until
  `models/best_xgboost.pkl` exists, then asserts the saved schema matches the
  code, that predictions come from the trained model, and that it agrees with
  the agronomic reference within a tolerance.

CI runs the suite on Python 3.10/3.11/3.12, then trains a small XGBoost model
and re-runs the full suite plus an end-to-end endpoint smoke test.

---

## Docker

```bash
docker build -t crop-yield-portal .
docker run --rm -p 8000:8000 crop-yield-portal

# Train inside the container so the image serves ML predictions
docker run --rm -v "$PWD/models:/app/models" crop-yield-portal \
  python train_model.py --n-samples 5000
```

The image listens on `8000` and serves both the frontend and the API.

---

## Deployment

| Target | What to deploy | Notes |
|--------|----------------|-------|
| Render / Railway / Fly.io | `app.py` | Set `PORT`; the app reads it from the environment |
| Docker host / Kubernetes | the image | Stateless; mount `models/` or train during build |
| GitHub Pages / Netlify / Vercel | `index.html`, `styles.css`, `app.js` | Static mode: results are labelled offline estimates |

`HOST` and `PORT` environment variables control the bind address, which makes
the service portable across PAAS providers that inject a port.

---

## Design notes

- **One feature implementation.** `features.py` is imported by both the trainer
  and the API, and `build_feature_frame()` returns columns in a fixed order that
  is persisted to `feature_columns.json` at training time. The API reindexes to
  the saved schema, so a model trained yesterday still loads today.
- **Honest fallbacks.** The portal always names the engine behind a number. A
  missing model degrades to a documented reference model instead of a 500.
- **Explicit static routes.** Only `index.html`, `styles.css` and `app.js` are
  served, so the API host does not expose the Python source.
- **ASCII-only CLI output.** Windows consoles using legacy code pages choke on
  emoji in `print()`; the scripts therefore print plain markers.
- **Pinned ranges, not pins.** `requirements.txt` uses lower bounds so CI picks
  up security patches; the CI matrix is what catches incompatibilities.
- **Progressive enhancement over graceful degradation.** Scroll reveals hide
  content only when `html.js` is present, so with JavaScript disabled every panel,
  chart bar and figure is visible at its final value instead of blank.

---

## Roadmap

- [x] Tabbed, animated, accessible frontend.
- [ ] Swap the synthetic generator for a real dataset (e.g. district-level
      records) behind a `--data` flag.
- [ ] Add rolling-origin cross-validation and residual plots.
- [ ] Persist prediction history to SQLite and expose `/history`.
- [ ] Add SHAP-based per-prediction explanations to the UI.
- [ ] Package the model as an ONNX graph for a lighter runtime.

---

## License

Released under the [MIT License](LICENSE).

Built with ❤️ for sustainable agriculture and data-driven farming.
