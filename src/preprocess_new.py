import pandas as pd
import re
import os


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_dataset1():
    """Kaggle Fake/True Dataset"""
    print("Loading Dataset 1: Kaggle Fake/True...")
    fake = pd.read_csv('data/Fake.csv')
    true = pd.read_csv('data/True.csv')

    fake['label'] = 'FAKE'
    true['label'] = 'REAL'

    df = pd.concat([fake, true], ignore_index=True)

    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']

    print(f"  ✅ Dataset 1: {len(df)} articles")
    return df[['content', 'label']]


def load_dataset2():
    """WELFake Dataset"""
    path = 'data/WELFake_Dataset.csv'
    if not os.path.exists(path):
        print("  ⚠️  WELFake_Dataset.csv not found, skipping...")
        return None

    print("Loading Dataset 2: WELFake...")
    df = pd.read_csv(path)

    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df['content'] = df['title'] + ' ' + df['text']

    # WELFake uses numbers: 0=FAKE, 1=REAL
    df['label'] = df['label'].map({0: 'FAKE', 1: 'REAL'})
    df = df.dropna(subset=['label'])

    print(f"  ✅ Dataset 2: {len(df)} articles")
    print(f"     FAKE: {len(df[df['label']=='FAKE'])} | REAL: {len(df[df['label']=='REAL'])}")
    return df[['content', 'label']]


def preprocess_data():
    print("=" * 60)
    print("LOADING ALL DATASETS")
    print("=" * 60)

    all_dfs = []

    # Load Dataset 1
    df1 = load_dataset1()
    all_dfs.append(df1)

    # Load Dataset 2
    df2 = load_dataset2()
    if df2 is not None:
        all_dfs.append(df2)

    # Combine
    print("\nCombining all datasets...")
    df = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates
    print("Removing duplicates...")
    before = len(df)
    df = df.drop_duplicates(subset=['content'])
    after = len(df)
    print(f"  Removed {before - after} duplicates")

    print(f"\nTotal before cleaning: {len(df)}")

    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['content'].apply(clean_text)

    # Remove short/empty entries
    df = df[df['cleaned_text'].str.strip().str.len() > 50]
    df = df.dropna(subset=['cleaned_text', 'label'])

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df[['cleaned_text', 'label']].to_csv('data/cleaned_news.csv', index=False)

    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total articles: {len(df)}")
    print(df['label'].value_counts())


def preprocess_text(text: str) -> str:
    """Preprocess a single news article string for prediction (used by web app)."""
    return clean_text(text)


if __name__ == "__main__":
    preprocess_data()