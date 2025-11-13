import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('Loading CSV...')
df = pd.read_csv('data/sample.csv')
print('Rows:', len(df))

# Count emotions
emotion_counts = {}
for labels in df['emotion_labels'].fillna('').str.split(','):
    for emotion in labels:
        emotion = emotion.strip()
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

print('Total distinct emotions:', len(emotion_counts))

counts = pd.DataFrame(list(emotion_counts.items()), columns=['emotion', 'count'])
counts = counts.sort_values('count', ascending=True)
print(counts.tail(10))

plt.figure(figsize=(10, 6))
bars = plt.barh(counts['emotion'], counts['count'], color='tab:blue')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{int(width):,d}', ha='left', va='center', fontsize=9)

plt.title('Emotion Distribution in Sample Data')
plt.xlabel('Count')
plt.ylabel('Emotion')
plt.tight_layout()
plt.savefig('scripts/sentiment_test_output.png', dpi=120)
print('Saved scripts/sentiment_test_output.png')
