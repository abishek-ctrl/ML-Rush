# ML-Rush — collection of ML projects

This repository, **ML-Rush**, is a personal archive for machine learning projects created during college. Each project lives in its own folder (for example `p01-house-price-prediction`) so you can iterate and revisit experiments later.

Current projects
- `p01-house-price-prediction/` — House price prediction (Ames Housing). See project README inside the folder for project-specific instructions.

Repository notes
- Data files should be referenced via environment variables (`.env`) and are gitignored. See each project's `.env.example` for defaults.
- Model artifacts (large binaries) should be stored separately (GitHub Releases, DVC, or cloud storage). `models/` is gitignored.
- Dockerfiles were removed to keep the repo lightweight — use the provided `requirements.txt` and `uvicorn` to run services locally.

Quickstart (example for `p01-house-price-prediction`)

1. Install dependencies

```bash
pip install -r p01-house-price-prediction/requirements.txt
```

2. Create a `.env` inside `p01-house-price-prediction/` by copying the example and adjust paths if needed:

```bash
cp p01-house-price-prediction/.env.example p01-house-price-prediction/.env
# edit .env to point TRAIN_PATH/TEST_PATH and MODEL_PATH
```

3. Run training (from `p01-house-price-prediction`)

```bash
python -m src.train
```

4. Run the API

```bash
uvicorn p01-house-price-prediction.api.main:app --reload --port 8000
```

Want to add another project? Create a new top-level folder `p02-...` with its own README, `requirements.txt`, and `.env.example`.
