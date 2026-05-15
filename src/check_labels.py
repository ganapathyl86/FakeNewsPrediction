import pandas as pd

df = pd.read_csv('data/cleaned_news.csv')

print("Label distribution:")
print(df['label'].value_counts())
print()
print("Sample FAKE news (label=0):")
print(df[df['label']==0]['cleaned_text'].iloc[0][:200])
print()
print("Sample REAL news (label=1):")
print(df[df['label']==1]['cleaned_text'].iloc[0][:200])
print()

# Check original files
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
print("FAKE.csv sample title:")
print(fake_df['title'].iloc[0])
print()
print("True.csv sample title:")
print(true_df['title'].iloc[0])
