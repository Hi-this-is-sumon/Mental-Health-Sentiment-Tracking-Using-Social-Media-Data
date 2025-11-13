from src.preprocess import clean_text
import joblib

data = joblib.load('models/sample_multilabel.joblib')
vectorizer = data['vectorizer']
model = data['model']
mlb = data['mlb']

text = 'i am feeling happy today'
cleaned = clean_text(text)
print('Cleaned:', cleaned)

vectorized = vectorizer.transform([cleaned])
print('Vectorized shape:', vectorized.shape)

probs = model.predict_proba(vectorized)[0]
print('Probs:', probs)

threshold = 0.5
preds_idx = [i for i, p in enumerate(probs) if p >= threshold]
print('Preds idx:', preds_idx)

if not preds_idx:
    preds_idx = [probs.argmax()]
    print('Using top emotion:', preds_idx)

preds = [mlb.classes_[i] for i in preds_idx]
print('Preds:', preds)

prob_map = {mlb.classes_[i]: float(probs[i]) for i in range(len(mlb.classes_))}
print('Prob map:', prob_map)
