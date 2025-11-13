import urllib.request

paths = ['/plot_sentiment', '/wordcloud']
for path in paths:
    url = 'http://127.0.0.1:5000' + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = r.read()
            print(path, 'status=', r.status, 'len=', len(data), 'ctype=', r.getheader('Content-Type'))
    except Exception as e:
        print(path, 'ERROR', e)
