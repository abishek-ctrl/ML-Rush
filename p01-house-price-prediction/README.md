# House Price Prediction — p01

Predict house sale prices using the Ames Housing dataset. This project demonstrates preprocessing, feature engineering, model development, evaluation, and an API for inference.

Important notes
- Place your CSVs at `data/train.csv` and `data/test.csv` (these paths are read from `.env`).
- Copy `.env.example` to `.env` and edit paths if needed. `.env` is gitignored.
- Model binaries (e.g., `house_price_model_tuned.pkl`) should be placed in `models/` (also gitignored). Consider using GitHub Releases for storing model artifacts.

Structure
- `data/` - Data files (gitignored)
- `models/` - Model files (gitignored)
- `src/` - Source modules: `preprocess.py`, `features.py`, `train.py`, `evaluate.py`
- `api/` - FastAPI app for predictions
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment variables

Usage

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Create `.env` from the example and adjust paths

```bash
cp .env.example .env
# edit .env (TRAIN_PATH, TEST_PATH, MODEL_PATH)
```

3. Train the model

```bash
python -m src.train
```

4. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```