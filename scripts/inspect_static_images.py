from PIL import Image
import os
paths=[r"static/plots/plot_sentiment.png", r"static/plots/wordcloud.png"]
for p in paths:
    print('---',p)
    if os.path.exists(p):
        print('size', os.path.getsize(p))
        try:
            img=Image.open(p)
            print('format', img.format, 'mode', img.mode, 'size', img.size, 'bbox', img.getbbox())
        except Exception as e:
            print('open error', e)
    else:
        print('MISSING')
