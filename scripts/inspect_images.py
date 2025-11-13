from PIL import Image

for name in ('scripts_plot_sentiment.png','scripts_wordcloud.png'):
    p=name
    try:
        im=Image.open(p)
        print(name, im.mode, im.size)
        alpha_present = 'A' in im.getbands()
        print(' alpha:', alpha_present)
        if alpha_present:
            bbox = im.split()[-1].getbbox()
            print(' non-transparent bbox:', bbox)
        else:
            extrema = im.convert('L').getextrema()
            print(' luminance extrema:', extrema)
    except Exception as e:
        print('error', name, e)
