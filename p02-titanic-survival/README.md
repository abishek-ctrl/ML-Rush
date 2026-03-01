# Titanic Survival Prediction — p02

Predict passenger survival probabilities on the Titanic. This project demonstrates modularized machine learning code, feature engineering, model training (Voting Ensemble), evaluation, and a FastAPI inference endpoint.

## Important Notes
- Place your raw CSVs at `data/raw/train.csv` and `data/raw/test.csv` (these paths are read from `.env`).
- Copy `.env.example` to `.env` and edit paths if needed. `.env` is gitignored.
- Model binaries (`titanic_model.pkl` and `titanic_threshold.pkl`) will be output to `models/` (also gitignored).

## Structure
- `data/` - Raw and processed data files (gitignored)
- `models/` - Model files (gitignored)
- `src/` - Source modules: `preprocess.py`, `features.py`, `train.py`, `evaluate.py`
- `api/` - FastAPI app for predictions (`main.py`)
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment variables

## Usage

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Setup environment**

```bash
cp .env.example .env
# edit .env if necessary
```

3. **Train the model**

```bash
python -m src.train
```

4. **Evaluate the model (generates plots)**

```bash
python -m src.evaluate
```

5. **Run the API**

```bash
uvicorn api.main:app --reload --port 8000
```
