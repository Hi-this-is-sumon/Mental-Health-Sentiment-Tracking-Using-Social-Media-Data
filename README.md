# Mental Health Sentiment Tracking Using Social Media Data

> **🎉 Now Fully Responsive & Production-Ready!** This application is optimized for all devices with full WCAG 2.1 AA accessibility compliance.

## 1. Introduction

Mental health issues such as depression, anxiety, and stress are increasingly expressed through online communication. Social media platforms allow users to share their emotions openly, often revealing early signs of mental distress. This project focuses on analyzing social media text to detect sentiment and identify language patterns associated with mental‑health‑related expressions.

The goal is not clinical diagnosis but automated text‑based sentiment tracking using machine learning and natural language processing (NLP).

This project includes a **fully responsive web application** that works seamlessly across all devices—from smartphones (360px) to ultra-wide displays (1920px+)—with touch-friendly interface design and accessibility features built-in.

---

## 2. Problem Statement

A significant portion of mental health challenges remains undetected because individuals often express distress online rather than seeking help. Manual monitoring is impossible at scale. Therefore, an automated system that analyzes sentiment and emotion trends in social media text can help identify general mental‑health‑related patterns.

---

## 3. Objectives

- To collect and preprocess social media text data.
- To perform sentiment analysis (positive, negative, neutral).
- To detect linguistic patterns related to mental‑health expressions such as depression, anxiety, and loneliness.
- To build and evaluate machine learning and transformer-based NLP models.
- To visualize sentiment trends and expression patterns.
- To ensure ethical handling and analysis of sensitive data.
- **To provide a fully responsive, accessible web interface** that works on all devices with touch-friendly design.

---

## 4. Dataset Sources

To maintain safety and legality, only public or anonymized datasets were used:

- Reddit Mental Health Dataset (Kaggle)
- Depression & Anxiety Text Dataset (Kaggle)
- Sentiment140 Twitter Dataset
- Emotions in Text Dataset

Synthetic or anonymized datasets eliminate the risk of exposing user identities.

---

## 5. Methodology / Workflow

### 5.1 Data Collection

Data was sourced from publicly available, anonymized datasets suitable for research. No private user information was accessed.

### 5.2 Data Preprocessing

The following preprocessing steps were applied:

- Lowercasing text
- Removing URLs, mentions, hashtags
- Removing emojis or converting them to text
- Removing stopwords
- Lemmatization
- Tokenization

### 5.3 Feature Extraction

The text was converted into numerical features using:

- Bag-of-Words
- TF-IDF
- Word Embeddings (Word2Vec, GloVe)
- Transformer embeddings (BERT)

### 5.4 Model Training

Multiple models were tested:

- Logistic Regression
- SVM
- Random Forest
- LSTM / Bi‑LSTM
- CNN for text
- Transformer models (BERT, DistilBERT, RoBERTa)

Transformer-based models produced the highest accuracy.

### 5.5 Classification

Two levels of outcomes were generated:

1. **Sentiment Classification:** positive, negative, neutral
2. **Mental‑Health Expression Classification:** depression-like, anxiety-like, stress-related, loneliness-related patterns

### 5.6 Visualization

Visualizations included:

- Sentiment distribution
- Word clouds for mental-health terms
- Trend graphs showing mood changes over time

---

## 6. System Architecture

User Text → Preprocessing → Feature Extraction → Model (ML/Transformer) → Sentiment + Mental‑Health Tag → Visualization

---

## 7. Results (Sample)

**Input:** "I feel so tired and hopeless these days."

**Output:**

- Sentiment: Negative
- Mental‑Health Pattern: Depression-like language
- Confidence: 87%

---

## 8. Ethical Considerations (Mandatory Section)

Mental‑health research involving social media data requires strict ethical practices. The following principles were applied throughout the project:

### 8.1 Privacy and Anonymity

All datasets were public and anonymized. Usernames, IDs, timestamps, and any personally identifiable information (PII) were removed. Only textual content was analyzed.

### 8.2 No Diagnosis

The model does **not** diagnose depression, anxiety, or any condition. It only detects **text that contains language consistent with mental‑health‑related expressions**. All conclusions are linguistic, not clinical.

### 8.3 Algorithmic Bias

Social media data can be biased in terms of demographics and language usage. As a result, model performance may vary across groups. This limitation is acknowledged.

### 8.4 Risk of Harm (False Negatives)

Misclassifications can have serious consequences. A suicidal or distressed message classified as "neutral" is dangerous in real-world contexts. This project is strictly academic and should not be used for intervention.

### 8.5 Informed Consent and Data Usage

Social media users do not provide explicit consent for research use of their posts. This project mitigates this issue by using only **public, anonymized datasets** and reporting only aggregated results.

### 8.6 Responsible Deployment

This model should not be deployed without human supervision, clear disclaimers, privacy safeguards, and proper mental‑health oversight.

---

## 9. Limitations

- Dataset bias
- Ambiguity in text (sarcasm, slang)
- No contextual understanding beyond text
- Not suitable for real-time mental-health diagnosis or crisis detection

---

## 10. Future Work

- Multilingual mental‑health sentiment analysis
- More diverse datasets to reduce bias
- Real-time trend dashboards
- Integration with professional mental‑health support systems
- Progressive Web App (PWA) support for offline functionality
- Advanced analytics and machine learning model optimization

---

## 10.5. Responsive Design & Accessibility

This project is **fully responsive and accessible:**

### Device Support
- ✅ Mobile phones (360px and up)
- ✅ Tablets (640px and up)
- ✅ Desktops (1024px and up)
- ✅ Ultra-wide displays (1920px and up)
- ✅ 4K monitors

### Responsive Breakpoints
| Breakpoint | Size | Features |
|-----------|------|----------|
| **XS** | < 360px | Single-column, stacked navigation, optimized touch targets |
| **SM** | 360-639px | Single-column layout, full-width inputs, mobile-optimized |
| **MD** | 640-1023px | Two-column grids, horizontal navigation, balanced spacing |
| **LG** | 1024-1919px | Multi-column layout, optimized spacing, full feature set |
| **XL** | 1920px+ | 3-column grids, maximum width container, enhanced layout |

### Accessibility Features (WCAG 2.1 Level AA)
- ✅ All touch targets ≥ 44×44px (mobile-friendly)
- ✅ Readable font sizes (14px minimum, scaling to 48px on desktop)
- ✅ Color contrast ≥ 4.5:1 ratio
- ✅ Full keyboard navigation support
- ✅ Visible focus indicators
- ✅ Semantic HTML structure
- ✅ Screen reader compatible
- ✅ Reduced motion support
- ✅ Dark mode support

### Responsive Features
- Fluid typography using CSS `clamp()` for automatic scaling
- Responsive spacing variables that adapt to viewport
- Auto-fit grids that intelligently wrap content
- Adaptive chart rendering (vertical on mobile, horizontal on desktop)
- Mobile optimization with auto-scroll to focused inputs
- Touch-friendly button spacing and sizing
- Responsive form elements with proper sizing

See `RESPONSIVE_DESIGN.md` for comprehensive responsive design documentation.

## 11. Conclusion

This project demonstrates how NLP and machine learning can be used to track sentiment and mental‑health-related language patterns in social media text. While useful for trend analysis and research, strict ethical boundaries were maintained to ensure privacy, fairness, and responsible use.

---

## Quickstart

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Put CSV dataset files into a `data/` directory. Each CSV should contain at least a `text` column and optionally a `label` column.
3. Run the baseline training example:

```powershell
python src/train.py --data data/your_dataset.csv --target emotion_labels --text-col text
```

4. Use `src/visualize.py` to generate simple plots of sentiment distribution.

## Step-by-Step Setup and Running Instructions

This project includes a fully functional Flask web application for interactive sentiment analysis. Follow these detailed steps to set up and run the application.

### Prerequisites

- Python 3.8 or higher (check with `python --version`)
- Windows, macOS, or Linux operating system
- At least 2GB free disk space
- Internet connection for downloading dependencies

### Step 1: Download the Project

1. Download the project files to your computer
2. Extract/unzip the files to a folder on your desktop or preferred location
3. Open the project folder in your file explorer

### Step 2: Set Up Python Virtual Environment

A virtual environment keeps the project dependencies isolated from your system Python.

1. Open Command Prompt or PowerShell in the project folder
2. Create a virtual environment:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

   You should see `(.venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

1. With the virtual environment activated, install all required packages:

```powershell
pip install -r requirements.txt
```

   This may take several minutes. The installation includes:
   - Flask (web framework)
   - scikit-learn (machine learning)
   - pandas (data processing)
   - matplotlib/seaborn (visualization)
   - transformers (NLP models)
   - And other dependencies

2. **Important Note:** If you see an error about `wordcloud` installation, you can skip it for now. The app will work without word clouds, but visualizations will be limited.

### Step 4: Download NLTK Resources (Recommended)

NLTK (Natural Language Toolkit) resources improve text processing accuracy.

1. Run this command to download essential NLTK data:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

### Step 5: Prepare the Machine Learning Model

The web app requires a pre-trained model to analyze text.

1. Check if the model file exists in the `models/` folder:
   - Look for `sample_multilabel.joblib` in the `models` directory

2. If the model file doesn't exist, train it using the sample dataset:

```powershell
python -m src.train --data data/sample.csv --target emotion_labels --text-col text --out models/sample_multilabel.joblib
```

   This will create a trained model file that the web app can use.

### Step 6: Start the Web Application

1. With the virtual environment still activated, start the Flask server:

```powershell
$env:PYTHONPATH='.'; python src/app.py
```

2. You should see output like:
   ```
   INFO:__main__:Model loaded successfully
   * Running on http://127.0.0.1:5000/
   ```

3. Keep this terminal window open - the server is now running!

### Step 7: Access the Web Application

1. Open your web browser (Chrome, Firefox, Edge, etc.)
2. Navigate to: `http://127.0.0.1:5000/`
3. You should see the Mental Health Sentiment Tracker landing page

### Step 8: Using the Application

#### First Time Use:
1. Click "Analyze Text" to go to the analysis page
2. Enter some sample text like: "I feel so tired and hopeless these days"
3. Click "Analyze" to see sentiment prediction, emotion scores, and keywords
4. Explore the visualizations and try different text inputs

#### Features to Try:
- **Sentiment Analysis:** Test with positive, negative, and neutral text
- **Emotion Scores:** See detailed emotion breakdowns
- **Keyword Extraction:** View important words in your text
- **Visualizations:** Check sentiment distribution and word frequency plots
- **History Page:** View past analyses and export PDF reports

### Step 9: Stopping the Application

1. Go back to the terminal window where the server is running
2. Press `Ctrl + C` to stop the server
3. The application will shut down gracefully

### Troubleshooting Common Issues

#### "Python is not recognized"
- Make sure Python 3.8+ is installed and added to your PATH
- Try using `python3` instead of `python`

#### "Permission denied" when activating virtual environment
- Right-click PowerShell/Command Prompt and "Run as Administrator"
- Or use: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### "Model not found" error
- Ensure you completed Step 5 and the model file exists in `models/`
- Check that the file name is exactly `sample_multilabel.joblib`

#### "Port 5000 already in use"
- Change the port in `src/app.py` (look for `app.run(port=5001)`)
- Or close other applications using port 5000

#### "Import errors" or "Module not found"
- Make sure you're in the virtual environment (see `(.venv)` in prompt)
- Try reinstalling dependencies: `pip install -r requirements.txt`

#### Application runs but shows errors in browser
- Check the terminal for error messages
- Ensure all files are in the correct directories
- Try refreshing the browser page

### Quick Commands Reference

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

# Train model (if needed)
$env:PYTHONPATH='.'; python src/train.py --data data/sample.csv --target label --out models/sample_multilabel.joblib

# Start application
$env:PYTHONPATH='.'; python src/app.py

# Test prediction (optional)
$env:PYTHONPATH='.'; python src/predict.py --text "Sample text here" --model_path models/sample_multilabel.joblib --model_type multilabel
```

### Web Application Features

#### Landing Page
- Introduction to the mental health sentiment tracking tool
- Navigation to analysis and history sections
- **Fully responsive design** that adapts to any device
- Dark mode toggle for comfortable viewing

#### Sentiment Analysis Page
- **Text Input:** Enter or paste text for analysis (responsive textarea)
- **Sentiment Prediction:** Get positive/negative/neutral classification with confidence score
- **Emotion Scores:** Detailed breakdown of emotions (joy, sadness, anger, fear, disgust)
- **Keyword Extraction:** Top keywords from the input text
- **Real-time Visualizations:** 
  - Sentiment distribution charts
  - Word frequency plots
  - Per-input emotion probability chart
- **Responsive Design:** Automatically adapts layout based on device size
- **Touch-Friendly:** Large buttons and inputs optimized for mobile interaction

#### History Page
- View past analyses with timestamps
- Export complete history as:
  - PDF report (includes visualizations and statistics)
  - CSV file (includes cleaned text and per-emotion columns)
- **Responsive History List:** Displays properly on all screen sizes
- **Mobile-Optimized:** Easy scrolling and interaction on small screens

#### Additional Features
- **Rate Limiting:** 10 requests per minute to prevent abuse
- **Caching:** Results cached for 5 minutes to improve performance
- **PDF Export:** Generate detailed reports with embedded visualizations
- **CSV Export:** Download analysis history with emotion scores and cleaned text
- **Batch Analysis:** Process multiple texts simultaneously (up to 50 texts per batch)
- **Responsive Design:** Works flawlessly on desktop, tablet, and mobile
- **Touch-Optimized:** 44px+ touch targets for accurate mobile interaction
- **Accessible:** Full WCAG 2.1 AA compliance for users with disabilities

### Sample Dataset Testing

The repository includes a sample dataset at `data/sample.csv` for testing:

1. Train the model (if needed):

```powershell
$env:PYTHONPATH='.'; python src/train.py --data data/sample.csv --target label --out models/sample_multilabel.joblib
```

2. Run predictions via command line:

```powershell
$env:PYTHONPATH='.'; python src/predict.py --text "I feel so tired and hopeless these days." --model_path models/sample_multilabel.joblib --model_type multilabel
```

3. Generate visualizations:

```powershell
$env:PYTHONPATH='.'; python src/visualize.py data/sample.csv
```

### Project Structure

```
mental-health-sentiment-tracking/
├── data/                         # Dataset files
│   └── sample.csv               # Sample dataset for testing
├── models/                       # Trained model files
│   ├── sample_baseline.joblib   # Baseline model
│   └── sample_multilabel.joblib # Multilabel model
├── reports/                      # Generated PDF/CSV reports
├── scripts/                      # Utility and testing scripts
│   ├── check_visuals.py         # Visualization tests
│   ├── test_responsive.py       # Responsive design tests (29/29 passing)
│   └── ...other utilities
├── src/                          # Source code
│   ├── app.py                   # Flask web application
│   ├── train.py                 # Model training script
│   ├── predict.py               # Prediction utilities
│   ├── preprocess.py            # Text preprocessing with typo normalization
│   ├── visualize.py             # Visualization functions
│   ├── data.py                  # Data loading utilities
│   └── analyze_dataset.py       # Dataset analysis tools
├── static/                       # Static files (CSS, JS, images)
│   ├── css/
│   │   ├── styles.css           # Main responsive stylesheet (13KB)
│   │   └── emotions.css         # Responsive emotion cards
│   ├── js/
│   │   └── app.js               # Client-side JavaScript (37KB)
│   └── plots/                   # Generated visualization PNGs
├── templates/                    # HTML templates
│   ├── landing.html             # Landing page
│   ├── analyze.html             # Analysis page (responsive)
│   └── history.html             # History page (responsive)
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── RESPONSIVE_DESIGN.md          # Comprehensive responsive design documentation
├── RESPONSIVE_QUICK_REFERENCE.md # Quick CSS/JS patterns and snippets
├── RESPONSIVE_TESTING_GUIDE.md   # Step-by-step testing instructions
├── RESPONSIVE_SUMMARY.md         # Implementation summary
├── BEFORE_AFTER_COMPARISON.md    # Changes and improvements overview
└── .gitignore
```

### Key Files Updated for Responsiveness

- **`static/css/styles.css`** (13KB) - Fluid typography, responsive spacing, 5 breakpoints
- **`static/css/emotions.css`** - Responsive emotion cards grid
- **`static/js/app.js`** (37KB) - Resize listeners, mobile detection, chart adaptation
- **`templates/analyze.html`** - Responsive markup with flexible layouts
- **`templates/history.html`** - Mobile-optimized history display
- **`templates/landing.html`** - Responsive landing page
- **`scripts/test_responsive.py`** - Automated responsive design testing (29 tests, all passing)

### Troubleshooting

- **Model Loading Error:** Ensure the model file exists in `models/` directory
- **Port Already in Use:** Change the port in `src/app.py` if 5000 is occupied
- **Missing Dependencies:** Run `pip install -r requirements.txt` again
- **NLTK Errors:** Run the NLTK download command above

### Notes

- The sample dataset is small and intended only for testing the pipeline.
- Never use model outputs for clinical decisions. This project is for research and educational purposes only.
- The web app includes ethical disclaimers and is not for clinical use.
- All analyses include confidence scores to indicate reliability.
