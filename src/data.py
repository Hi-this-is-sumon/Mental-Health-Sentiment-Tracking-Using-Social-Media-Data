import pandas as pd
from typing import Optional

from preprocess import clean_text


class SimpleDatasetLoader:
    """Loads CSV files with a `text` column and optional label column.

    Features:
    - configurable text column name (default: 'text')
    - configurable label column name (default: None)
    - supports comma-separated multi-label strings (returns list of labels in 'labels')

    Usage:
        loader = SimpleDatasetLoader('data/my.csv', text_col='message', label_col='sentiment')
        df = loader.load()
    """

    def __init__(self, path: str, text_col: str = 'text', label_col: str = None):
        self.path = path
        self.text_col = text_col
        self.label_col = label_col

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        if self.text_col not in df.columns:
            raise ValueError(f"CSV must contain a '{self.text_col}' column")

        # Clean text into a new column `text_clean` so downstream code has a consistent field
        df['text_clean'] = df[self.text_col].astype(str).apply(clean_text)

        # If a label column is provided, normalize it
        if self.label_col and self.label_col in df.columns:
            # If values look comma-separated, convert to list for multi-label
            def _parse_labels(v):
                if pd.isna(v):
                    return []
                if isinstance(v, list):
                    return v
                s = str(v)
                # common separators: comma or pipe
                if ',' in s:
                    return [x.strip() for x in s.split(',') if x.strip()]
                if '|' in s:
                    return [x.strip() for x in s.split('|') if x.strip()]
                # single label
                return [s.strip()]

            df['labels'] = df[self.label_col].apply(_parse_labels)

        return df


def load_multiple(paths: list) -> pd.DataFrame:
    dfs = [SimpleDatasetLoader(p).load() for p in paths]
    return pd.concat(dfs, ignore_index=True)
