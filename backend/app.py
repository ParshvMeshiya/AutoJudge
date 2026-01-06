import sys
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scipy.sparse import hstack, csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_engineering import (
    text_length,
    word_count,
    math_symbol_count,
    keyword_features,
    constraint_features,
    input_structure_features,
    algorithmic_depth,
    DIFFICULTY_KEYWORDS,
    CONSTRAINT_FEATURE_NAMES,
    STRUCTURE_FEATURE_NAMES
)

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="AutoJudge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --------------------------------------------------
# Load models
# --------------------------------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
clf = joblib.load(os.path.join(MODEL_DIR, "classifier_logreg.pkl"))
regressor = joblib.load(os.path.join(MODEL_DIR, "regressor_rf.pkl"))

# --------------------------------------------------
# Request schema
# --------------------------------------------------
class PredictRequest(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip().lower()
    if not text:
        return {"error": "Empty input"}

    # ---------- TF-IDF ----------
    X_tfidf = tfidf.transform([text])

    # ---------- Manual features (MUST match training order) ----------
    manual_features = []

    manual_features.extend([
        text_length(text),
        word_count(text),
        math_symbol_count(text),
        algorithmic_depth(text),
    ])

    manual_features.extend(keyword_features(text))
    manual_features.extend(constraint_features(text))
    manual_features.extend(input_structure_features(text))

    X_dense = csr_matrix(np.array(manual_features).reshape(1, -1))

    # ---------- Final feature ----------
    X_final = hstack([X_tfidf, X_dense])

    # ---------- Classification ----------
    probs = clf.predict_proba(X_final)[0]
    pred_idx = probs.argmax()
    difficulty = clf.classes_[pred_idx]
    confidence = probs[pred_idx]

    # ---------- Score calibration ----------
    ranges = {
        "easy": (1.0, 4.0),
        "medium": (4.0, 7.0),
        "hard": (7.0, 10.0)
    }

    lo, hi = ranges[difficulty]
    base_score = lo + confidence * (hi - lo)

    try:
        reg_score = regressor.predict(X_final)[0]
        score = 0.8 * base_score + 0.2 * reg_score
    except:
        score = base_score

    return {
        "difficulty": difficulty,
        "confidence": round(float(confidence), 2),
        "problem_score": round(float(score), 2)
    }
