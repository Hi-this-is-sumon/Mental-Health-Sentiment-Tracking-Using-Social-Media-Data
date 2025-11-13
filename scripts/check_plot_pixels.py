from PIL import Image
import numpy as np
paths=['static/plots/plot_sentiment.png','static/plots/wordcloud.png']
for p in paths:
    try:
        img=Image.open(p).convert('RGBA')
        arr=np.array(img)
        total=arr.shape[0]*arr.shape[1]
        alpha=arr[:,:,3]
        rgb=arr[:,:,:3]
        nonwhite=((alpha>10) & ~(np.all(rgb>250,axis=2)))
        count_nonwhite=int(nonwhite.sum())
        uniq=np.unique(arr.reshape(-1,4),axis=0)
        print(p, 'size', img.size, 'nonwhite_pixels', count_nonwhite, 'total_pixels', total, 'unique_colors', len(uniq))
    except Exception as e:
        print('ERROR',p,e)
