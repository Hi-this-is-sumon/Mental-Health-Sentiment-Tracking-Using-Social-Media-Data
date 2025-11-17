import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    pass

from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')  # Set backend to non-GUI
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean_text
from visualize import plot_sentiment_distribution, generate_wordcloud
from data import SimpleDatasetLoader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64
import logging
import threading

app = Flask(__name__, template_folder='../templates', static_folder='../static')
# Secret key for session signing; set via environment variable in production
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("App initialized. Using simple session-based authentication.")

# Helper function to resolve file paths relative to project root
def get_project_path(relative_path):
    """Resolve a relative path from the project root (parent of src/)."""
    # __file__ is src/app.py
    # Go up 2 levels: src/app.py -> src -> (project root)
    app_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(app_dir)  # project root
    full_path = os.path.join(project_root, relative_path)
    
    # If path doesn't exist and we're in a nested structure (Render), try going up another level
    if not os.path.exists(full_path):
        parent_root = os.path.dirname(project_root)
        full_path = os.path.join(parent_root, relative_path)
    
    return full_path

from functools import wraps

def require_login(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Simple hardcoded credentials for demo (no database needed)
        valid_users = {
            'user': 'password123',
            'admin': 'admin123',
            'demo': 'demo'
        }
        
        if username in valid_users and valid_users[username] == password:
            session['user'] = {'name': username, 'id': username}
            logger.info(f"User {username} logged in")
            return redirect(url_for('analyze'))
        else:
            error = 'Invalid username or password'
            logger.warning(f"Failed login attempt for user: {username}")
    
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))



# Configure caching and rate limiting
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)



# Load the trained model (prefer multilabel model if available)
vectorizer = None
model = None
mlb = None
for candidate in ('models/sample_multilabel.joblib', 'models/sample_baseline.joblib'):
    try:
        model_data = joblib.load(get_project_path(candidate))
        vectorizer = model_data.get('vectorizer')
        model = model_data.get('model')
        mlb = model_data.get('mlb')
        logger.info(f"Model loaded successfully from {candidate}")
        break
    except FileNotFoundError:
        logger.warning(f"Model file not found: {candidate}")
    except Exception as e:
        logger.error(f"Error loading model {candidate}: {e}")

if model is None or vectorizer is None:
    logger.error("No usable model found. Please train the model first and save to models/*.joblib")


# Pre-warm visualization cache in background so first browser requests are fast
def _prewarm_visuals():
    try:
        with app.test_client() as c:
            logger.info("Pre-warming /plot_sentiment cache")
            c.get('/plot_sentiment')
            logger.info("Pre-warming /wordcloud cache")
            c.get('/wordcloud')
            logger.info("Pre-warm complete")
    except Exception as e:
        logger.warning(f"Pre-warm visuals failed: {e}")


try:
    t = threading.Thread(target=_prewarm_visuals, daemon=True)
    t.start()
except Exception:
    logger.info("Skipping pre-warm thread")

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/analyze')
@require_login
def analyze():
    return render_template('analyze.html')

@app.route('/history')
@require_login
def history():
    return render_template('history.html')

def validate_text_input(data):
    """Validate and extract text from request data."""
    if not data:
        logger.warning("No JSON data provided in request")
        return None, jsonify({'error': 'No JSON data provided'}), 400

    text = data.get('text', '').strip()
    if not text:
        logger.warning("Empty text provided in request")
        return None, jsonify({'error': 'No text provided'}), 400

    if len(text) > 10000:  # Reasonable limit for text input
        logger.warning(f"Text too long: {len(text)} characters")
        return None, jsonify({'error': 'Text too long (max 10000 characters)'}), 400

    return text, None, None

def perform_sentiment_analysis(text):
    """Perform sentiment analysis on the given text."""
    global vectorizer, model, mlb
    if not model or not vectorizer:
        # Try to load model if not loaded
        for candidate in ('models/sample_multilabel.joblib', 'models/sample_baseline.joblib'):
            try:
                model_data = joblib.load(get_project_path(candidate))
                vectorizer = model_data.get('vectorizer')
                model = model_data.get('model')
                mlb = model_data.get('mlb')
                logger.info(f"Model loaded successfully from {candidate}")
                break
            except FileNotFoundError:
                logger.warning(f"Model file not found: {candidate}")
            except Exception as e:
                logger.error(f"Error loading model {candidate}: {e}")

    if not model or not vectorizer:
        logger.error("Model not loaded")
        return None, jsonify({'error': 'Model not loaded'}), 500

    try:
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])

        # If model is multilabel (has mlb), return list of predicted labels and per-label probs
        if mlb is not None:
            probs = model.predict_proba(vectorized)[0]
            threshold = 0.5
            preds_idx = [i for i, p in enumerate(probs) if p >= threshold]
            if not preds_idx:
                # If no emotions above threshold, include the top emotion
                preds_idx = [probs.argmax()]
            preds = [mlb.classes_[i] for i in preds_idx]
            prob_map = {mlb.classes_[i]: float(probs[i]) for i in range(len(mlb.classes_))}
            return {
                'prediction': preds,
                'probabilities': prob_map
            }, None, None

        prediction = model.predict(vectorized)[0]
        confidence = float(max(model.predict_proba(vectorized)[0]))

        return {
            'prediction': prediction,
            'confidence': confidence
        }, None, None

    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return None, jsonify({'error': 'Analysis failed'}), 500

@limiter.limit("10 per minute")
@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")

    data = request.get_json()
    text, error_response, status_code = validate_text_input(data)
    if error_response:
        return error_response, status_code

    result, error_response, status_code = perform_sentiment_analysis(text)
    if error_response:
        return error_response, status_code

    logger.info(f"Prediction completed successfully for text length: {len(text)}")
    return jsonify(result)

@app.route('/batch_predict', methods=['POST'])
@limiter.limit("5 per minute")
def batch_predict():
    """Process multiple texts for sentiment analysis in a single request."""
    logger.info("Received batch prediction request")

    data = request.get_json()
    if not data or 'texts' not in data:
        logger.warning("No texts provided in batch request")
        return jsonify({'error': 'No texts provided'}), 400

    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("Invalid texts format in batch request")
        return jsonify({'error': 'texts must be a non-empty list'}), 400

    if len(texts) > 50:  # Limit batch size
        logger.warning(f"Batch size too large: {len(texts)}")
        return jsonify({'error': 'Maximum 50 texts per batch'}), 400

    results = []
    errors = []

    for i, text in enumerate(texts):
        try:
            # Validate individual text
            if not isinstance(text, str) or not text.strip():
                errors.append({'index': i, 'error': 'Invalid text'})
                continue

            if len(text) > 10000:
                errors.append({'index': i, 'error': 'Text too long'})
                continue

            # Perform analysis
            cleaned = clean_text(text)
            vectorized = vectorizer.transform([cleaned])

            if mlb is not None:
                probs = model.predict_proba(vectorized)[0]
                threshold = 0.5
                preds_idx = [j for j, p in enumerate(probs) if p >= threshold]
                preds = [mlb.classes_[j] for j in preds_idx]
                prob_map = {mlb.classes_[j]: float(probs[j]) for j in range(len(mlb.classes_))}
                results.append({
                    'index': i,
                    'prediction': preds,
                    'probabilities': prob_map
                })
            else:
                prediction = model.predict(vectorized)[0]
                confidence = float(max(model.predict_proba(vectorized)[0]))
                results.append({
                    'index': i,
                    'prediction': prediction,
                    'confidence': confidence
                })

        except Exception as e:
            logger.error(f"Error processing text at index {i}: {e}")
            errors.append({'index': i, 'error': 'Analysis failed'})

    response = {'results': results}
    if errors:
        response['errors'] = errors

    logger.info(f"Batch prediction completed: {len(results)} successful, {len(errors)} errors")
    return jsonify(response)

@app.route('/emotion_scores', methods=['POST'])
def emotion_scores():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    # For multilabel model return per-label probabilities as emotion scores
    if mlb is not None:
        probs = model.predict_proba(vectorized)[0]
        prob_map = {mlb.classes_[i]: float(probs[i]) for i in range(len(mlb.classes_))}
        dominant = max(prob_map, key=prob_map.get) if prob_map else None
        return jsonify({
            'emotions': prob_map,
            'dominant_emotion': dominant
        })

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    # Map predictions to emotion scores (simplified for demo)
    emotion_map = {
        'negative': {'sadness': 0.7, 'anger': 0.3, 'fear': 0.4, 'disgust': 0.2, 'joy': 0.1},
        'positive': {'joy': 0.8, 'sadness': 0.1, 'anger': 0.0, 'fear': 0.0, 'disgust': 0.0},
        'neutral': {'joy': 0.3, 'sadness': 0.3, 'anger': 0.1, 'fear': 0.1, 'disgust': 0.1}
    }

    base_emotions = emotion_map.get(prediction.lower(), emotion_map['neutral'])
    # Adjust based on confidence
    confidence = float(max(probabilities))
    adjusted_emotions = {k: min(1.0, v * (0.5 + confidence)) for k, v in base_emotions.items()}

    return jsonify({
        'emotions': adjusted_emotions,
        'dominant_emotion': max(adjusted_emotions, key=adjusted_emotions.get)
    })

@app.route('/keywords', methods=['POST'])
def keywords():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    words = cleaned.split()

    # Simple keyword extraction (remove stopwords, get top frequency)
    from nltk.corpus import stopwords
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]

    # Get top 10 keywords by frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    return jsonify({
        'keywords': [{'word': word, 'frequency': freq} for word, freq in top_keywords]
    })

@cache.cached(timeout=300)  # Cache for 5 minutes
@app.route('/plot_sentiment')
def plot_sentiment():
    try:
        # If a previously-generated static plot exists and is recent, serve it directly for speed
        plots_dir = os.path.join(app.static_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        static_path = os.path.join(plots_dir, 'plot_sentiment.png')
        if os.path.exists(static_path):
            # If the file is younger than cache timeout (300s), return it
            mtime = os.path.getmtime(static_path)
            if (pd.Timestamp.now().timestamp() - mtime) < 300:
                return send_file(static_path, mimetype='image/png')
        print("Loading data for emotion distribution plot...")
        df = pd.read_csv(get_project_path('data/sample.csv'))

        # Sample a smaller subset for faster processing (max 10,000 rows for speed)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
            print(f"Sampled to {len(df)} rows for faster processing")

        print(f"Data loaded: {len(df)} rows")

        if df.empty:
            print("DataFrame is empty")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=16)
            plt.title('Emotion Distribution - No Data')
        else:
            print(f"Data columns: {df.columns.tolist()}")

            # Split multi-label emotions and count occurrences
            emotion_counts = {}
            for labels in df['emotion_labels'].fillna('').str.split(','):
                for emotion in labels:
                    emotion = emotion.strip()
                    if emotion:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Convert to DataFrame for plotting
            counts = pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count'])
            # Ensure counts are integers (avoid float formatting issues later)
            counts['count'] = pd.to_numeric(counts['count'], errors='coerce').fillna(0).astype(int)
            counts = counts.sort_values('count', ascending=True)  # For horizontal bar plot

            if counts.empty:
                print("Counts is empty")
                plt.text(0.5, 0.5, 'No sentiment data available', ha='center', va='center', fontsize=16)
                plt.title('Sentiment Distribution - No Data')
            else:
                print("Creating barplot...")
                # Create horizontal bar plot (dark-friendly: transparent background and light text)
                from matplotlib import rc_context
                # Use dark text/colors on a white background so plots are visible
                with rc_context({'figure.facecolor': 'white',
                                 'axes.facecolor': 'white',
                                 'axes.edgecolor': '#111111',
                                 'axes.labelcolor': '#111111',
                                 'xtick.color': '#111111',
                                 'ytick.color': '#111111',
                                 'text.color': '#111111'}):
                    plt.figure(figsize=(10, 6), facecolor='white')
                    bars = plt.barh(counts['emotion'], counts['count'], color='#2b8cc4')

                # Add value labels on the bars (use integer formatting)
                for i, bar in enumerate(bars):
                    width = int(round(bar.get_width()))
                    plt.text(width, bar.get_y() + bar.get_height()/2,
                             f'{width:,}',
                             ha='left', va='center', fontsize=10)

                plt.title('Emotion Distribution in Sample Data')
                plt.xlabel('Count')
                plt.ylabel('Emotion')

                # Adjust layout
                plt.tight_layout()
                print("Saving plot...")

                img = io.BytesIO()
                # Save with white background and lower DPI to reduce file size
                plt.savefig(img, format='png', bbox_inches='tight', dpi=100, transparent=False)
                img.seek(0)

                # Crop any large transparent margins using PIL to avoid empty top/bottom space
                try:
                    from PIL import Image
                    pil = Image.open(img)
                    bbox = pil.getbbox()
                    if bbox:
                        pil = pil.crop(bbox)
                    out = io.BytesIO()
                    pil.save(out, format='PNG')
                    out.seek(0)
                    # Persist the cropped image
                    try:
                        with open(static_path, 'wb') as f:
                            f.write(out.getbuffer())
                    except Exception as e:
                        logger.warning(f"Failed to write static plot file: {e}")
                    plt.close()
                    print("Plot saved successfully (cropped)")
                    return send_file(io.BytesIO(out.getbuffer()), mimetype='image/png')
                except Exception as e:
                    # If PIL is not available or cropping fails, fallback to original image
                    logger.debug(f"Image cropping failed, returning original image: {e}")
                    img.seek(0)
                    try:
                        with open(static_path, 'wb') as f:
                            f.write(img.getbuffer())
                    except Exception as e:
                        logger.warning(f"Failed to write static plot file: {e}")
                    plt.close()
                    print("Plot saved successfully (uncropped)")
                    return send_file(io.BytesIO(img.getbuffer()), mimetype='image/png')
    except Exception as e:
        print(f"Error generating sentiment plot: {e}")
        import traceback
        traceback.print_exc()

        # Create error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', fontsize=12, color='red', wrap=True)
        plt.title('Sentiment Distribution - Error')
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', transparent=False)
        img.seek(0)
        try:
            with open(static_path, 'wb') as f:
                f.write(img.getbuffer())
        except Exception as e:
            logger.warning(f"Failed to write static error plot file: {e}")
        plt.close()
        return send_file(io.BytesIO(img.getbuffer()), mimetype='image/png')

@app.route('/model_info')
def model_info():
    """Return model metadata including type (multilabel/single), classes, and configuration."""
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500
        
    info = {
        'model_type': 'multilabel' if mlb else 'single-label',
        'classes': mlb.classes_.tolist() if mlb else model.classes_.tolist(),
        'vectorizer_config': {
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
            'vocab_size': len(vectorizer.vocabulary_)
        }
    }
    
    # Add metadata if present (from new training)
    model_data = joblib.load(get_project_path('models/sample_multilabel.joblib'))
    if 'meta' in model_data:
        info.update(model_data['meta'])
        
    return jsonify(info)


@app.route('/export_history_csv', methods=['POST'])
def export_history_csv():
    """Return a CSV file for the provided history JSON (includes per-emotion columns)."""
    try:
        data = request.get_json() or {}
        history = data.get('history') or []
        if not history:
            return jsonify({'error': 'No history provided'}), 400

        # Collect all emotion keys
        emotions = set()
        rows = []
        for entry in history:
            probs = entry.get('probabilities') or {}
            emotions.update(probs.keys())

        emotions = sorted(emotions)

        # Build CSV using StringIO for proper escaping
        import csv
        out = io.StringIO()
        writer = csv.writer(out)

        header = ['timestamp', 'text', 'text_clean', 'prediction', 'confidence'] + emotions
        writer.writerow(header)

        for entry in history:
            timestamp = entry.get('timestamp', '')
            text = entry.get('text', '')
            # Compute cleaned text server-side if not provided
            if 'text_clean' in entry and entry.get('text_clean'):
                text_clean = entry.get('text_clean')
            else:
                try:
                    text_clean = clean_text(text)
                except Exception:
                    text_clean = ''
            prediction = ','.join(entry['prediction']) if isinstance(entry.get('prediction'), list) else entry.get('prediction', '')
            confidence = entry.get('confidence', '')
            probs = entry.get('probabilities') or {}
            # Normalize probabilities as floats (keep empty if missing)
            row_probs = [probs.get(e, '') for e in emotions]
            row = [timestamp, text, text_clean, prediction, confidence] + row_probs
            writer.writerow(row)

        out.seek(0)
        return send_file(io.BytesIO(out.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='analysis_history.csv')

    except Exception as e:
        logger.error(f"Failed to generate CSV: {e}")
        return jsonify({'error': 'Failed to generate CSV'}), 500

def generate_word_frequency_plot(df, max_words=20, title='Top Words in Sample Data'):
    """Generate a word frequency bar chart from the message column."""
    import re
    from collections import Counter

    def clean_word(word: str) -> str:
        word = re.sub(r"[^\w\s]", "", word)
        word = re.sub(r"\d+", "", word)
        return word.lower().strip()

    # Try to use NLTK stopwords if available, otherwise fall back to small set
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'i', 'me', 'my', 'you', 'your', "'s"
    }
    try:
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords as _sw
        stop_words = set(w.lower() for w in _sw.words('english')) | stop_words
    except Exception:
        # NLTK not available; use fallback stopwords above
        pass

    # Simple contractions expansion for common forms
    contractions = {
        "i'm": 'i am', "can't": 'cannot', "won't": 'will not', "don't": 'do not',
        "it's": 'it is', "that's": 'that is', "i've": 'i have', "you're": 'you are',
        "we're": 'we are', "they're": 'they are', "isn't": 'is not', "aren't": 'are not',
        "didn't": 'did not', "couldn't": 'could not', "wouldn't": 'would not'
    }

    words = []
    # Support both 'message' and 'text' column names in datasets
    text_col = 'message' if 'message' in df.columns else ('text' if 'text' in df.columns else None)
    if text_col is None:
        raise ValueError("No 'message' or 'text' column found in dataframe for word frequency generation")
    for msg in df[text_col].astype(str):
        # expand simple contractions first
        lower = msg.lower()
        for c, rep in contractions.items():
            lower = lower.replace(c, rep)
        for raw in lower.split():
            w = clean_word(raw)
            # filter out stopwords, tokens with length < 3, and single quotes
            if not w or w in stop_words or len(w) < 3 or all(ch == "'" for ch in w):
                continue
            words.append(w)

    if not words:
        raise ValueError('No words extracted from messages')

    word_freq = Counter(words)
    top_words = word_freq.most_common(max_words)

    from matplotlib import rc_context
    # Use dark text on white background for consistent visibility
    with rc_context({'figure.facecolor': 'white',
                     'axes.facecolor': 'white',
                     'axes.edgecolor': '#111111',
                     'axes.labelcolor': '#111111',
                     'xtick.color': '#111111',
                     'ytick.color': '#111111',
                     'text.color': '#111111'}):
        plt.figure(figsize=(11, 5), facecolor='white')
        bars = plt.bar([w[0] for w in top_words], [w[1] for w in top_words], color='#2b8cc4')
        for bar in bars:
            h = int(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:,}", ha='center', va='bottom')

        plt.title(title)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100, transparent=False)
    img.seek(0)

    # Crop transparent whitespace with PIL to avoid large empty margins
    try:
        from PIL import Image
        pil = Image.open(img)
        bbox = pil.getbbox()
        if bbox:
            pil = pil.crop(bbox)
        out = io.BytesIO()
        pil.save(out, format='PNG')
        out.seek(0)
        plt.close()
        return out
    except Exception as e:
        # If cropping fails, return original image
        img.seek(0)
        plt.close()
        return img

@app.route('/wordcloud')
def wordcloud():
    """Return a PNG bar chart of the top words in the `message` column."""
    try:
        # If a previously-generated static plot exists and is recent, serve it directly for speed
        plots_dir = os.path.join(app.static_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        static_path = os.path.join(plots_dir, 'wordcloud.png')
        if os.path.exists(static_path):
            mtime = os.path.getmtime(static_path)
            if (pd.Timestamp.now().timestamp() - mtime) < 300:
                return send_file(static_path, mimetype='image/png')
        # Read data and sample if too large
        df = pd.read_csv(get_project_path('data/sample.csv'))
        if df.shape[0] > 50000:
            df = df.sample(n=50000, random_state=42)

        img = generate_word_frequency_plot(df)
        # Persist image for faster subsequent requests
        try:
            with open(static_path, 'wb') as f:
                f.write(img.getbuffer())
        except Exception as e:
            logger.warning(f"Failed to write static wordcloud file: {e}")
        return send_file(io.BytesIO(img.getbuffer()), mimetype='image/png')
    except Exception as e:
        logger.error(f"Error generating word frequency plot: {e}")
        # Create error plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error generating visualization: {str(e)}', ha='center', va='center', fontsize=12, wrap=True)
        plt.title('Word Frequency - Error')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', transparent=False)
        img.seek(0)
        try:
            with open(static_path, 'wb') as f:
                f.write(img.getbuffer())
        except Exception as e:
            logger.warning(f"Failed to write static wordcloud error file: {e}")
        plt.close()
        return send_file(io.BytesIO(img.getbuffer()), mimetype='image/png')

def generate_pdf(history, filename, is_full_history=False):
    """Generate a starter doc/content/styles tuple for building a PDF.

    This helper returns the ReportLab document object, an initially-empty content
    list, and the style sheet to be used by the caller.
    """
    # Create PDF buffer and document object
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, filename=filename)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center alignment
    )

    content = []

    # Different titles and formats based on export type
    if is_full_history:
        content.append(Paragraph("Mental Health Analysis History", title_style))
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    else:
        content.append(Paragraph("Emotion Analysis Report", title_style))
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))

    return doc, content, styles

@app.route('/export_history_pdf', methods=['POST'])
def export_history_pdf():
    try:
        data = request.get_json()
        history = data.get('history', [])

        if not history:
            return jsonify({'error': 'No history data provided'}), 400

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
        )

        normal_style = styles['Normal']
        normal_style.spaceAfter = 12

        # Build PDF content
        content = []

        # Title
        content.append(Paragraph("Mental Health Sentiment Analysis History Report", title_style))
        content.append(Spacer(1, 12))

        # Disclaimer
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b> This analysis is for research and educational purposes only.
        It does not provide medical diagnosis, treatment, or professional advice. Results are based
        on text analysis and should not be used for clinical decisions. If you're experiencing
        mental health concerns, please consult a qualified healthcare professional.
        """
        content.append(Paragraph(disclaimer_text, normal_style))
        content.append(Spacer(1, 20))

        # Summary Statistics
        content.append(Paragraph("Analysis Summary", subtitle_style))

        # Calculate summary stats
        total_analyses = len(history)
        
        # Handle multi-label predictions
        prediction_counts = {}
        total_probability_sum = 0
        total_emotions = 0
        
        for entry in history:
            if isinstance(entry['prediction'], list):
                # Multi-label case
                for emotion in entry['prediction']:
                    prediction_counts[emotion] = prediction_counts.get(emotion, 0) + 1
                # Get average probability for detected emotions
                probabilities = entry.get('probabilities', {})
                detected_probs = [probabilities[emotion] for emotion in entry['prediction'] if emotion in probabilities]
                if detected_probs:
                    total_probability_sum += sum(detected_probs)
                    total_emotions += len(detected_probs)
            else:
                # Single-label case
                prediction_counts[entry['prediction']] = prediction_counts.get(entry['prediction'], 0) + 1
                if 'confidence' in entry:
                    total_probability_sum += entry['confidence']
                    total_emotions += 1

        avg_confidence = total_probability_sum / total_emotions if total_emotions > 0 else 0

        summary_data = [
            ['Total Analyses', str(total_analyses)],
            ['Average Confidence', f"{(avg_confidence * 100):.1f}%"],
            ['Date Range', f"{pd.to_datetime(history[-1]['timestamp']).strftime('%Y-%m-%d')} to {pd.to_datetime(history[0]['timestamp']).strftime('%Y-%m-%d')}"]
        ]

        for pred, count in prediction_counts.items():
            summary_data.append([f"{pred.title()} Count", str(count)])

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(summary_table)
        content.append(Spacer(1, 20))

        # Individual Analyses
        content.append(Paragraph("Individual Analysis Results", subtitle_style))
        content.append(Spacer(1, 12))

        for i, entry in enumerate(history, 1):
            content.append(Paragraph(f"Analysis #{i}", styles['Heading3']))
            content.append(Spacer(1, 6))

            # Format predictions and scores
            if isinstance(entry['prediction'], list):
                # Multi-label case
                emotions_text = ', '.join(entry['prediction']) if entry['prediction'] else 'No emotions detected'
                probabilities = entry.get('probabilities', {})
                scores_text = ', '.join(f"{emotion}: {probabilities.get(emotion, 0)*100:.1f}%" 
                                      for emotion in entry['prediction']) if entry['prediction'] else 'N/A'
            else:
                # Single-label case
                emotions_text = entry['prediction']
                scores_text = f"{entry.get('confidence', 0)*100:.1f}%"

            # Different formats for single analysis vs history
            if 'Export Results' in doc.filename:
                analysis_data = [
                    ['Analysis Date & Time', pd.to_datetime(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')],
                    ['Input Text', entry['text'][:200] + '...' if len(entry['text']) > 200 else entry['text']],
                    ['Detected Emotions', emotions_text],
                    ['Confidence Scores', scores_text]
                ]
            else:
                # For history export, use a more compact format
                analysis_data = [
                    ['Entry #', str(i)],
                    ['Date', pd.to_datetime(entry['timestamp']).strftime('%Y-%m-%d %H:%M')],
                    ['Input', entry['text'][:150] + '...' if len(entry['text']) > 150 else entry['text']],
                    ['Emotions', emotions_text]
            ]

            analysis_table = Table(analysis_data, colWidths=[2*inch, 4*inch])
            analysis_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            content.append(analysis_table)
            content.append(Spacer(1, 12))

        # Add visualizations section
        content.append(Paragraph("Data Visualizations", subtitle_style))
        content.append(Spacer(1, 12))

        # Add visualizations if available (prefer static cropped images when present)
        try:
            # Get emotion distribution plot (handles multi-label by counting emotion_labels)
            sentiment_img_data = None
            df_plot = pd.read_csv(get_project_path('data/sample.csv'))
            emotion_counts = {}
            for labels in df_plot['emotion_labels'].fillna('').astype(str).str.split(','):
                for emo in labels:
                    emo = emo.strip()
                    if emo:
                        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

            if not emotion_counts:
                raise ValueError('No emotion labels found in dataset')

            counts_df = pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count']).sort_values('count', ascending=True)
            plt.figure(figsize=(8, 6))
            bars = plt.barh(counts_df['emotion'], counts_df['count'], color='#4a90e2')
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{int(bar.get_width()):,}", va='center', ha='left')
            plt.title('Emotion Distribution in Sample Data')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(sentiment_img_data := io.BytesIO(), format='png', bbox_inches='tight', dpi=100)
            plt.close()
            sentiment_img_data.seek(0)

            # Prefer static file if available (these are cropped when created by /plot_sentiment)
            plots_dir = os.path.join(app.static_folder, 'plots')
            static_sentiment = os.path.join(plots_dir, 'plot_sentiment.png')
            content.append(Paragraph("1. Emotion Distribution", styles['Heading3']))
            content.append(Paragraph("Distribution of labeled emotions in the dataset.", normal_style))
            if os.path.exists(static_sentiment):
                content.append(Image(static_sentiment, width=6*inch, height=4.5*inch))
            else:
                content.append(Image(sentiment_img_data, width=6*inch, height=4.5*inch))
            content.append(Spacer(1, 20))

        except Exception as e:
            print(f"Could not add emotion distribution plot to PDF: {e}")
            content.append(Paragraph("Note: Emotion distribution chart could not be generated.", styles['Italic']))

        try:
            # Word frequency plot (using 'message' column)
            wordcloud_img_data = io.BytesIO()
            df_msg = pd.read_csv(get_project_path('data/sample.csv'))
            import re
            from collections import Counter

            def clean_word(w):
                w = re.sub(r"[^\w\s]", "", w)
                w = re.sub(r"\d+", "", w)
                return w.lower().strip()

            # Support both 'message' and 'text' column names in datasets
            text_col = 'message' if 'message' in df_msg.columns else ('text' if 'text' in df_msg.columns else None)
            if text_col is None:
                raise ValueError("No 'message' or 'text' column found in dataset for word frequency generation")

            words = []
            for msg in df_msg[text_col].astype(str):
                for raw in msg.split():
                    cw = clean_word(raw)
                    if cw:
                        words.append(cw)

            top_words = Counter(words).most_common(20)

            plt.figure(figsize=(10, 6))
            bars = plt.bar([w[0] for w in top_words], [w[1] for w in top_words], color='skyblue')
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f"{int(bar.get_height()):,}", ha='center', va='bottom')
            plt.title('Top 20 Words in Sample Data')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(wordcloud_img_data, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            wordcloud_img_data.seek(0)

            # Prefer static wordcloud if available (cropped by /wordcloud)
            static_word = os.path.join(plots_dir, 'wordcloud.png')
            content.append(Paragraph("2. Word Frequency Analysis", styles['Heading3']))
            content.append(Paragraph("This chart displays the most frequently occurring words in our sample dataset.", normal_style))
            if os.path.exists(static_word):
                content.append(Image(static_word, width=6*inch, height=4.5*inch))
            else:
                content.append(Image(wordcloud_img_data, width=6*inch, height=4.5*inch))
            content.append(Spacer(1, 20))

        except Exception as e:
            print(f"Could not add word frequency plot to PDF: {e}")
            content.append(Paragraph("Note: Word frequency chart could not be generated.", styles['Italic']))

        # Footer
        footer_text = """
        This comprehensive history report was generated by the Mental Health Sentiment Tracker.
        For more information, visit our research documentation.
        """
        content.append(Paragraph(footer_text, styles['Italic']))

        # Build and save PDF
        doc.build(content)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='mental_health_sentiment_history_report.pdf'
        )

    except Exception as e:
        print(f"Error generating history PDF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate history PDF report'}), 500

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()
        text = data.get('text', '')
        prediction = data.get('prediction', '')
        confidence = data.get('confidence', 0)

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
        )

        normal_style = styles['Normal']
        normal_style.spaceAfter = 12

        # Build PDF content
        content = []

        # Title
        content.append(Paragraph("Single Message Sentiment Analysis", title_style))
        content.append(Spacer(1, 12))

        # Disclaimer
        disclaimer_text = """
        <b>IMPORTANT DISCLAIMER:</b> This analysis is for research and educational purposes only.
        It does not provide medical diagnosis, treatment, or professional advice. Results are based
        on text analysis and should not be used for clinical decisions. If you're experiencing
        mental health concerns, please consult a qualified healthcare professional.
        """
        content.append(Paragraph(disclaimer_text, normal_style))
        content.append(Spacer(1, 20))

        # Analysis Details
        content.append(Paragraph("Analysis Details", subtitle_style))

        # Handle multilabel predictions
        if isinstance(prediction, list):
            predicted_emotions = ', '.join(prediction)
            confidence_text = "See emotion scores below"
        else:
            predicted_emotions = prediction
            confidence_text = f"{(confidence * 100):.1f}%"

        # Create table with analysis results
        analysis_data = [
            ['Input Text', text[:200] + '...' if len(text) > 200 else text],
            ['Predicted Emotions', predicted_emotions],
            ['Confidence Score', confidence_text],
            ['Analysis Date', pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]

        table = Table(analysis_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(table)
        content.append(Spacer(1, 20))

        # Interpretation
        content.append(Paragraph("Interpretation", subtitle_style))
        if isinstance(prediction, list):
            interpretation_text = f"""
            Based on linguistic patterns in your text, our model detected elements commonly associated
            with the following emotions: {', '.join(prediction)}. The emotion scores indicate how certain
            the model is about each prediction.
            """
        else:
            interpretation_text = f"""
            Based on linguistic patterns in your text, our model detected elements commonly associated
            with {prediction.lower()} sentiment. The confidence score of {(confidence * 100):.1f}%
            indicates how certain the model is about this prediction.
            """
        content.append(Paragraph(interpretation_text, normal_style))
        content.append(Spacer(1, 12))

        # Gentle Suggestion
        content.append(Paragraph("Gentle Suggestion", subtitle_style))
        suggestion_text = """
        Remember to take care of yourself. If you're feeling overwhelmed, consider talking to a
        trusted friend or professional. Small acts of self-care can make a big difference.
        """
        content.append(Paragraph(suggestion_text, normal_style))
        content.append(Spacer(1, 20))

        # Add visualizations section
        content.append(Paragraph("Data Visualizations", subtitle_style))
        content.append(Spacer(1, 12))

        # Add visualizations if available
        try:
            # Get emotion distribution plot (handles multi-label by counting emotion_labels)
            sentiment_img_data = io.BytesIO()
            df_plot = pd.read_csv(get_project_path('data/sample.csv'))
            emotion_counts = {}
            for labels in df_plot['emotion_labels'].fillna('').astype(str).str.split(','):
                for emo in labels:
                    emo = emo.strip()
                    if emo:
                        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

            if not emotion_counts:
                raise ValueError('No emotion labels found in dataset')

            counts_df = pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count']).sort_values('count', ascending=True)
            plt.figure(figsize=(8, 6))
            bars = plt.barh(counts_df['emotion'], counts_df['count'], color='#4a90e2')
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{int(bar.get_width()):,}", va='center', ha='left')
            plt.title('Emotion Distribution in Sample Data')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(sentiment_img_data := io.BytesIO(), format='png', bbox_inches='tight', dpi=100)
            plt.close()
            sentiment_img_data.seek(0)

            plots_dir = os.path.join(app.static_folder, 'plots')
            static_sentiment = os.path.join(plots_dir, 'plot_sentiment.png')
            content.append(Paragraph("1. Emotion Distribution", styles['Heading3']))
            content.append(Paragraph("This chart shows the distribution of different emotion categories in our sample dataset.", normal_style))
            if os.path.exists(static_sentiment):
                content.append(Image(static_sentiment, width=6*inch, height=4.5*inch))
            else:
                content.append(Image(sentiment_img_data, width=6*inch, height=4.5*inch))
            content.append(Spacer(1, 20))

        except Exception as e:
            print(f"Could not add emotion distribution plot to PDF: {e}")
            content.append(Paragraph("Note: Emotion distribution chart could not be generated.", styles['Italic']))

        try:
            # Get word frequency plot
            wordcloud_img_data = io.BytesIO()
            # Load dataset and support either 'message' or 'text' as the text column
            df = pd.read_csv(get_project_path('data/sample.csv'))
            text_col = 'message' if 'message' in df.columns else ('text' if 'text' in df.columns else None)
            if text_col is None:
                raise ValueError("No 'message' or 'text' column found in dataset for word frequency generation")

            # Prefer cleaned text if available, otherwise use raw text column
            if 'text_clean' in df.columns:
                text = ' '.join(df['text_clean'].astype(str))
            else:
                text = ' '.join(df[text_col].astype(str))
            words = text.split()
            word_freq = {}
            for word in words:
                word = word.lower().strip('.,!?')
                if word:
                    word_freq[word] = word_freq.get(word, 0) + 1
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

            plt.figure(figsize=(10, 6))
            plt.bar([w[0] for w in top_words], [w[1] for w in top_words], color='skyblue')
            plt.title('Top 20 Words in Sample Data')
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(wordcloud_img_data, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            wordcloud_img_data.seek(0)

            static_word = os.path.join(plots_dir, 'wordcloud.png')
            content.append(Paragraph("2. Word Frequency Analysis", styles['Heading3']))
            content.append(Paragraph("This chart displays the most frequently occurring words in our sample dataset, helping identify common themes and patterns.", normal_style))
            if os.path.exists(static_word):
                content.append(Image(static_word, width=6*inch, height=4.5*inch))
            else:
                content.append(Image(wordcloud_img_data, width=6*inch, height=4.5*inch))
            content.append(Spacer(1, 20))

        except Exception as e:
            print(f"Could not add word frequency plot to PDF: {e}")
            content.append(Paragraph("Note: Word frequency chart could not be generated.", styles['Italic']))

        # Footer
        footer_text = """
        This report was generated by the Mental Health Sentiment Tracker.
        For more information, visit our research documentation.
        """
        content.append(Paragraph(footer_text, styles['Italic']))

        # Build and save PDF
        doc.build(content)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='single_message_analysis.pdf'
        )

    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to generate PDF report'}), 500

if __name__ == '__main__':
    app.run(debug=True)
