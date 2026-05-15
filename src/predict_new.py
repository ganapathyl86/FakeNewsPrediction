import joblib
import re
import os


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_models():
    # FIX 1: Use absolute path — works from any directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    print("Loading models...")
    try:
        tfidf        = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer_new.pkl'))
        lr_model     = joblib.load(os.path.join(models_dir, 'logistic_regression_new.pkl'))
        dt_model     = joblib.load(os.path.join(models_dir, 'decision_tree_new.pkl'))
        rf_model     = joblib.load(os.path.join(models_dir, 'random_forest_new.pkl'))
        voting_model = joblib.load(os.path.join(models_dir, 'voting_classifier_new.pkl'))
        print("✅ All models loaded!\n")
        return tfidf, lr_model, dt_model, rf_model, voting_model
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("   Run train_new.py first to generate model files.")
        exit(1)


def predict_all(text, tfidf, models_dict):
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    # Individual model predictions
    for name, model in models_dict.items():
        if name == 'Soft Voting Classifier':
            continue
        pred = model.predict(X)[0]

        if hasattr(model, 'predict_proba'):
            proba   = model.predict_proba(X)[0]
            classes = list(model.classes_)
            fake_p  = proba[classes.index('FAKE')] * 100
            real_p  = proba[classes.index('REAL')] * 100
            print(f"{name:<25} → {pred:<6} (FAKE:{fake_p:.1f}% | REAL:{real_p:.1f}%)")
        else:
            print(f"{name:<25} → {pred}")

    # FIX 2: Show soft voting averaged probabilities — not vote count
    voting_model = models_dict['Soft Voting Classifier']
    voting_pred  = voting_model.predict(X)[0]

    if hasattr(voting_model, 'predict_proba'):
        v_proba    = voting_model.predict_proba(X)[0]
        v_classes  = list(voting_model.classes_)
        v_fake_p   = v_proba[v_classes.index('FAKE')] * 100
        v_real_p   = v_proba[v_classes.index('REAL')] * 100
        print("=" * 60)
        print(f"Soft Voting (avg prob) → FAKE:{v_fake_p:.1f}% | REAL:{v_real_p:.1f}%")

    print("=" * 60)
    print(f"🎯 FINAL VERDICT : {voting_pred}")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("    FAKE NEWS DETECTION SYSTEM")
    print("=" * 60)

    tfidf, lr, dt, rf, voting = load_models()

    models_dict = {
        'Logistic Regression'  : lr,
        'Decision Tree'        : dt,
        'Random Forest'        : rf,
        'Soft Voting Classifier': voting
    }

    print("Type 'quit' to exit\n")

    while True:
        print("-" * 60)
        text = input("📰 Enter news text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if len(text) < 10:
            print("⚠️  Please enter more text!")
            continue

        predict_all(text, tfidf, models_dict)


if __name__ == "__main__":
    main()