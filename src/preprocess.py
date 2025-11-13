import re
import string
from typing import List

try:
    import nltk
except Exception:
    nltk = None


def _safe_nltk_setup():
    """Attempt to use NLTK resources; if unavailable provide safe fallbacks.

    This avoids blocking network downloads at import time in offline or sandboxed
    environments. If NLTK tokenizers/lemmatizer are missing we fall back to
    a simple regex tokenizer and no-op lemmatizer.
    """
    tokenizer = None
    lemmatizer = None
    stopwords_set = set()
    if nltk is not None:
        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            # try to load resources (may raise if not already installed)
            stopwords_set = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            # ensure the punkt tokenizer is available before using word_tokenize
            try:
                # require both the punkt tokenizer and punkt_tab data (some NLTK
                # installs separate the 'punkt_tab' resource). If either is
                # missing, fall back to the regex tokenizer to avoid runtime
                # LookupError.
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('tokenizers/punkt_tab/english')
                tokenizer = nltk.word_tokenize
            except Exception:
                tokenizer = lambda s: re.findall(r"\w+", s)
        except Exception:
            # resources missing; fall back
            tokenizer = lambda s: re.findall(r"\w+", s)
            lemmatizer = None
            stopwords_set = set()
    else:
        tokenizer = lambda s: re.findall(r"\w+", s)
    return stopwords_set, lemmatizer, tokenizer


STOPWORDS, LEMMATIZER, TOKENIZE = _safe_nltk_setup()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_HASHTAG = re.compile(r"[@#]\w+")
EMOJI_PATTERN = re.compile(r"[\U00010000-\U0010ffff]")


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Basic cleaning: lowercase, remove URLs, mentions, hashtags, punctuation, extra whitespace.

    Args:
        text: raw text string
        remove_stopwords: whether to remove stopwords
        lemmatize: whether to lemmatize tokens (only if a lemmatizer is available)

    Returns:
        cleaned string
    """
    if not isinstance(text, str):
        return ""
    txt = text.lower()

    # Quick normalization for common typos to improve model robustness for short user input.
    # Keep this list small and conservative to avoid changing legitimate words.
    def normalize_common_typos(s: str) -> str:
        replacements = {
            "\bfelling\b": "feeling",
            "\bfren\b": "friend",
            "\bteh\b": "the",
            "\brecieve\b": "receive",
            "\bdefinately\b": "definitely",
            "\bcant\b": "can't",
            "\bwont\b": "won't",
            "\bim\b": "i'm",
            "\bur\b": "your",
        }
        for pat, rep in replacements.items():
            s = re.sub(pat, rep, s)
        return s

    txt = normalize_common_typos(txt)
    txt = URL_PATTERN.sub("", txt)
    txt = MENTION_HASHTAG.sub("", txt)
    txt = EMOJI_PATTERN.sub("", txt)
    txt = txt.translate(str.maketrans("", "", string.punctuation))

    tokens: List[str] = TOKENIZE(txt)
    cleaned_tokens: List[str] = []
    for t in tokens:
        if remove_stopwords and t in STOPWORDS:
            continue
        if lemmatize and LEMMATIZER is not None:
            try:
                t = LEMMATIZER.lemmatize(t)
            except Exception:
                pass
        cleaned_tokens.append(t)
    return " ".join(cleaned_tokens).strip()


def preprocess_series(texts: List[str]) -> List[str]:
    """Apply clean_text to a list/iterable of texts."""
    return [clean_text(t) for t in texts]
