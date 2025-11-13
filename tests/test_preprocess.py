from src.preprocess import clean_text


def test_clean_text_basic():
    s = 'Check this out! https://example.com @user #hashtag'
    cleaned = clean_text(s)
    assert 'http' not in cleaned
    assert '@' not in cleaned
    assert '#' not in cleaned
    assert cleaned == cleaned.lower()


if __name__ == '__main__':
    test_clean_text_basic()
    print('preprocess test passed')
