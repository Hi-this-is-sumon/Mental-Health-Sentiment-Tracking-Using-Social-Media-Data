import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_sentiment_distribution(df: pd.DataFrame, label_col: str = 'label'):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in DataFrame")
    counts = df[label_col].value_counts().reset_index()
    counts.columns = [label_col, 'count']
    plt.figure(figsize=(6,4))
    sns.barplot(x=label_col, y='count', data=counts)
    plt.title('Sentiment / Label Distribution')
    plt.tight_layout()
    plt.show()


def generate_wordcloud(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label', label_filter: str = None):
    """Generate word cloud for mental-health terms, optionally filtered by label."""
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not in DataFrame")
    if label_filter and label_col in df.columns:
        df = df[df[label_col] == label_filter]
    text = ' '.join(df[text_col].astype(str))
    # Simple text-based word cloud without WordCloud library
    words = text.split()
    word_freq = {}
    for word in words:
        word = word.lower().strip('.,!?')
        if word:
            word_freq[word] = word_freq.get(word, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    plt.figure(figsize=(10, 5))
    plt.bar([w[0] for w in top_words], [w[1] for w in top_words])
    plt.title(f'Top Words for {label_filter or "All"} Texts')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    # quick demo: expects a CSV path in env or current dir
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        plot_sentiment_distribution(df, 'label')
    else:
        print('Usage: python src/visualize.py path/to/dataset.csv')
