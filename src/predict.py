import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Import `clean_text` with fallbacks so this script works whether run as
# a module (e.g. `python -m src.predict`) or as a script (e.g. `python src/predict.py`).
try:
    # Preferred when running tests or installed as a package
    from src.preprocess import clean_text
except Exception:
    try:
        # Preferred when running from the `src` directory (as in app.py)
        from preprocess import clean_text
    except Exception:
        # Last-resort relative import for module execution
        from .preprocess import clean_text


def predict_baseline(text: str, model_path: str):
    """Predict using baseline TF-IDF + LogisticRegression model."""
    model_data = joblib.load(model_path)
    vectorizer = model_data['vectorizer']
    model = model_data['model']
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    # Support for multilabel models saved with 'mlb'
    mlb = model_data.get('mlb')
    if mlb is not None:
        # OneVsRestClassifier with LogisticRegression supports predict_proba
        probs = model.predict_proba(vectorized)[0]
        # threshold to consider positive label
        threshold = 0.5
        preds_idx = [i for i, p in enumerate(probs) if p >= threshold]
        preds = [mlb.classes_[i] for i in preds_idx]
        # build dict of label->prob
        prob_map = {mlb.classes_[i]: float(probs[i]) for i in range(len(mlb.classes_))}
        return preds, prob_map

    # single-label fallback
    pred = model.predict(vectorized)[0]
    probs = model.predict_proba(vectorized)[0]
    confidence = float(max(probs))
    return pred, confidence


def predict(text: str, model_path: str, model_type: str = 'baseline'):
    """Unified predict function."""
    if model_type == 'baseline':
        return predict_baseline(text, model_path)
    else:
        raise ValueError("Unsupported model type")


def _cli():
    parser = argparse.ArgumentParser(description='Predict sentiment/mental-health pattern from text')
    parser.add_argument('--text', required=True, help='Text to analyze')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--model_type', default='baseline', choices=['baseline'], help='Model type')
    args = parser.parse_args()
    pred, conf = predict(args.text, args.model_path, args.model_type)
    print(f"Prediction: {pred}")
    # `conf` may be a numeric confidence or a dict of label->probabilities for multilabel models.
    if isinstance(conf, dict):
        print("Probabilities:")
        for k, v in conf.items():
            try:
                print(f"  {k}: {float(v):.4f}")
            except Exception:
                print(f"  {k}: {v}")
    else:
        try:
            print(f"Confidence: {float(conf):.2f}")
        except Exception:
            print(f"Confidence: {conf}")


if __name__ == '__main__':
    _cli()
