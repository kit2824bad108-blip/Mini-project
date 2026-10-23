# Contributing

Thanks for considering a contribution to the Crop Yield Prediction Portal.

## Development setup

```bash
git clone https://github.com/kit2824bad108-blip/crop-yield-prediction-portal.git
cd crop-yield-prediction-portal

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
python train_model.py            # produces models/best_xgboost.pkl
```

## Running the app

```bash
python app.py                    # serves the frontend and the API on :8000
```

Open <http://127.0.0.1:8000> for the portal, or <http://127.0.0.1:8000/docs> for
the interactive API documentation.

## Tests

```bash
pytest
```

The suite runs without a trained model (it exercises the documented fallback
engine) and adds extra assertions automatically once
`models/best_xgboost.pkl` exists.

## Project conventions

- **Single source of truth for features.** All feature construction lives in
  `features.py` and is imported by both `train_model.py` and `app.py`. Never
  duplicate the feature logic — that is how train/serve skew creeps in.
- **Keep the frontend dependency-free.** Plain HTML, CSS and vanilla JS only.
  No build step, no bundler, no CDN.
- **ASCII-only console output.** The CLI prints ASCII markers (`[OK]`, `>`)
  rather than emoji so it works on Windows consoles using legacy code pages.
- **Update the schema consciously.** Changing `FEATURE_COLUMNS` invalidates any
  previously saved model — retrain and commit nothing, since `models/` is
  git-ignored.

## Pull requests

1. Fork the repository, then branch from `master`.
2. Keep changes focused; one concern per pull request.
3. Run `pytest` before opening the PR.
4. Describe the *why* in the PR description, and note any behaviour change in
   the API contract.
