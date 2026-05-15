import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_models():
    print("=" * 60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 60)

    # [1/6] Load directly from True.csv and Fake.csv
    print("\n[1/6] Loading dataset...")
    true_df = pd.read_csv('data/True.csv')
    fake_df = pd.read_csv('data/Fake.csv')

    # ✅ CORRECT labels: Real=1, Fake=0
    true_df['label'] = 1
    fake_df['label'] = 0

    df = pd.concat([true_df[['text', 'label']],
                    fake_df[['text', 'label']]],
                   ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✓ Total articles: {len(df)}")
    print(f"  REAL (1): {len(df[df['label'] == 1])}")
    print(f"  FAKE (0): {len(df[df['label'] == 0])}")

    # [2/6] Clean text
    print("\n[2/6] Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    print("✓ Text cleaning done!")

    # [3/6] Features and labels
    print("\n[3/6] Train/Test split (80:20)...")
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training samples : {len(X_train)}")
    print(f"✓ Testing samples  : {len(X_test)}")

    # [4/6] TF-IDF
    print("\n[4/6] TF-IDF Vectorization...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)
    print(f"✓ Features: {X_train_tfidf.shape[1]}")
    print("✓ Using unigram + bigram features")

    # [5/6] Training models
    print("\n[5/6] Training models...")

    print("\n  Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(X_test_tfidf))
    print(f"  ✓ Logistic Regression : {lr_acc * 100:.2f}%")

    print("\n  Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_model.fit(X_train_tfidf, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test_tfidf))
    print(f"  ✓ Decision Tree       : {dt_acc * 100:.2f}%")

    print("\n  Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_tfidf, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_tfidf))
    print(f"  ✓ Random Forest       : {rf_acc * 100:.2f}%")

    # ✅ FIXED: pass already fitted models to VotingClassifier
    print("\n  Training Soft Voting Classifier...")
    voting_model = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('dt', dt_model),
            ('rf', rf_model)
        ],
        voting='soft'
    )
    voting_model.fit(X_train_tfidf, y_train)
    voting_pred = voting_model.predict(X_test_tfidf)
    voting_acc  = accuracy_score(y_test, voting_pred)
    print(f"  ✓ Soft Voting Classifier : {voting_acc * 100:.2f}%")

    # Results
    print("\n" + "=" * 60)
    print("FINAL ACCURACY COMPARISON")
    print("=" * 60)
    print(f"Logistic Regression    : {lr_acc * 100:.2f}%")
    print(f"Decision Tree          : {dt_acc * 100:.2f}%")
    print(f"Random Forest          : {rf_acc * 100:.2f}%")
    print(f"Soft Voting Classifier : {voting_acc * 100:.2f}%")
    print("=" * 60)

    print("\nClassification Report (Soft Voting Classifier):")
    print(classification_report(y_test, voting_pred,
          target_names=['FAKE (0)', 'REAL (1)']))

    # [6/6] Save models
    print("\n[6/6] Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf,        'models/tfidf_vectorizer_new.pkl')
    joblib.dump(lr_model,     'models/logistic_regression_new.pkl')
    joblib.dump(dt_model,     'models/decision_tree_new.pkl')
    joblib.dump(rf_model,     'models/random_forest_new.pkl')
    joblib.dump(voting_model, 'models/voting_classifier_new.pkl')
    print("✅ All models saved to models/ folder!")

    # Sanity check
    print("\n--- SANITY CHECK ---")
    print("Classes:", voting_model.classes_)
    fake_test = tfidf.transform(['anonymous sources deep state conspiracy whistleblowers silenced share before deleted'])
    real_test = tfidf.transform(['federal reserve held interest rates steady inflation progress unanimous decision'])
    print("Fake article → pred:", voting_model.predict(fake_test),
          "| proba:", voting_model.predict_proba(fake_test))
    print("Real article → pred:", voting_model.predict(real_test),
          "| proba:", voting_model.predict_proba(real_test))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    train_models()