import argparse
import os
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


def train_baseline(data_path: str, target: str = 'label', text_col: str = 'text', model_out: Optional[str] = None):
    """Train a baseline TF-IDF + LogisticRegression model.

    Supports single-label (string labels) and multi-label (comma-separated labels) targets.
    If multi-label is detected, uses MultiLabelBinarizer + OneVsRestClassifier.
    """
    df = pd.read_csv(data_path)

    if text_col not in df.columns:
        raise ValueError(f"Input CSV must have a '{text_col}' column")
    df[text_col] = df[text_col].astype(str)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in CSV")

    # Prepare X
    X = df[text_col].astype(str)

    # Detect multi-label: presence of comma or pipe in some values
    sample_vals = df[target].dropna().astype(str).head(50).tolist()
    is_multilabel = any(',' in s or '|' in s for s in sample_vals)

    vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_t = vect.fit_transform(X)

    if is_multilabel:
        print('Detected multi-label target. Using MultiLabelBinarizer + OneVsRestClassifier')
        mlb = MultiLabelBinarizer()
        y_lists = df[target].fillna('').astype(str).apply(lambda s: [x.strip() for x in s.split(',') if x.strip()])
        Y = mlb.fit_transform(y_lists)

        X_train, X_test, y_train, y_test = train_test_split(X_t, Y, test_size=0.2, random_state=42)

        base = LogisticRegression(max_iter=1000)
        clf = OneVsRestClassifier(base)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        # classification_report supports multilabel indicator format
        print('Hamming loss:', hamming_loss(y_test, y_pred))
        try:
            print(classification_report(y_test, y_pred, target_names=mlb.classes_))
        except Exception:
            # fallback if shapes mismatch
            print('Could not generate full classification_report for multilabel case')

        if model_out:
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            joblib.dump({'vectorizer': vect, 'model': clf, 'mlb': mlb}, model_out)
            print(f"Saved multilabel model to {model_out}")
    else:
        print('Detected single-label target. Using single-label LogisticRegression')
        y = df[target].astype(str)
        X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))

        if model_out:
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            joblib.dump({'vectorizer': vect, 'model': clf}, model_out)
            print(f"Saved model to {model_out}")


def _cli():
    parser = argparse.ArgumentParser(description='Train a baseline TF-IDF + LogisticRegression model')
    parser.add_argument('--data', required=True, help='Path to CSV with text and label columns')
    parser.add_argument('--target', default='label', help='Name of the target/label column')
    parser.add_argument('--text-col', default='text', help='Name of the text column')
    parser.add_argument('--out', default='models/baseline.joblib', help='Path to save trained model')
    args = parser.parse_args()
    train_baseline(args.data, args.target, args.text_col, args.out)


if __name__ == '__main__':
    _cli()
