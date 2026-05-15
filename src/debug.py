import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Load model
tfidf = joblib.load('models/tfidf_vectorizer_new.pkl')
rf = joblib.load('models/random_forest_new.pkl')

# Load data
df = pd.read_csv('data/cleaned_news.csv')

print("="*60)
print("1. DATASET CHECK")
print("="*60)
print(f"Total: {len(df)}")
print(f"Label 0 (FAKE): {len(df[df['label']==0])}")
print(f"Label 1 (REAL): {len(df[df['label']==1])}")

print("\n" + "="*60)
print("2. TESTING WITH 10 ACTUAL SAMPLES FROM DATASET")
print("="*60)

# Take 5 FAKE and 5 REAL from actual data
fake_samples = df[df['label']==0].head(5)
real_samples = df[df['label']==1].head(5)

correct = 0
total = 0

for _, row in fake_samples.iterrows():
    X = tfidf.transform([row['cleaned_text']])
    pred = rf.predict(X)[0]
    result = "✅ CORRECT" if pred == 0 else "❌ WRONG"
    print(f"[FAKE] Predicted: {'FAKE' if pred==0 else 'REAL'} {result}")
    if pred == 0:
        correct += 1
    total += 1

for _, row in real_samples.iterrows():
    X = tfidf.transform([row['cleaned_text']])
    pred = rf.predict(X)[0]
    result = "✅ CORRECT" if pred == 1 else "❌ WRONG"
    print(f"[REAL] Predicted: {'REAL' if pred==1 else 'FAKE'} {result}")
    if pred == 1:
        correct += 1
    total += 1

print(f"\nAccuracy on samples: {correct}/{total}")

print("\n" + "="*60)
print("3. TESTING CUSTOM NEWS (with same preprocessing)")
print("="*60)

custom_tests = [
    ("NASA James Webb telescope captured images of Carina Nebula revealing young stars", "REAL"),
    ("Donald Trump sends embarrassing message disturbing new year eve", "FAKE"),
    ("Federal Reserve raised interest rates fighting inflation economy", "REAL"),
    ("Obama arrested treason espionage FBI charges share before deleted", "FAKE"),
]

for text, expected in custom_tests:
    cleaned = remove_stopwords(clean_text(text))
    X = tfidf.transform([cleaned])
    pred = rf.predict(X)[0]
    predicted = 'FAKE' if pred == 0 else 'REAL'
    result = "✅" if predicted == expected else "❌"
    print(f"{result} Expected: {expected} | Got: {predicted}")
    print(f"   Text: {text[:60]}...")
    print()

print("\n" + "="*60)
print("4. CHECKING IF MODEL WAS RETRAINED AFTER PREPROCESSING FIX")
print("="*60)
import os
import time

model_time = os.path.getmtime('models/random_forest_new.pkl')
data_time = os.path.getmtime('data/cleaned_news.csv')

print(f"cleaned_news.csv last modified: {time.ctime(data_time)}")
print(f"random_forest_new.pkl last modified: {time.ctime(model_time)}")

if model_time < data_time:
    print("\n⚠️  WARNING: Model is OLDER than data!")
    print("    You need to retrain! Run: python src/train_new.py")
else:
    print("\n✅ Model is newer than data - training is up to date")
