from src import app as application

client = application.app.test_client()
for path in ['/plot_sentiment', '/wordcloud']:
    resp = client.get(path)
    data = resp.data
    print(path, 'status=', resp.status_code, 'len=', len(data), 'ctype=', resp.content_type)
    # Optionally save to disk for inspection
    open('scripts'+path.replace('/','_')+'.png','wb').write(data)
print('Saved files to scripts_plot_sentiment.png and scripts_wordcloud.png')
