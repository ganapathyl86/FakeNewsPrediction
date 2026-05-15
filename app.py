from flask import Flask, render_template, request
import joblib
import re
import os

app = Flask(__name__)

# ---------- Text cleaning ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Model loading ----------
def load_models():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    tfidf_path = os.path.join(models_dir, 'tfidf_vectorizer_new.pkl')
    lr_path    = os.path.join(models_dir, 'logistic_regression_new.pkl')
    dt_path    = os.path.join(models_dir, 'decision_tree_new.pkl')
    rf_path    = os.path.join(models_dir, 'random_forest_new.pkl')
    vote_path  = os.path.join(models_dir, 'voting_classifier_new.pkl')

    print("Loading models from:", models_dir)
    tfidf        = joblib.load(tfidf_path)
    lr_model     = joblib.load(lr_path)
    dt_model     = joblib.load(dt_path)
    rf_model     = joblib.load(rf_path)
    voting_model = joblib.load(vote_path)
    print("All models loaded!")
    print("Classes:", voting_model.classes_)

    return tfidf, lr_model, dt_model, rf_model, voting_model

# Load once when app starts
tfidf, lr_model, dt_model, rf_model, voting_model = load_models()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error      = None
    input_text = ""
    fake_prob  = None
    real_prob  = None

    if request.method == "POST":
        input_text = request.form.get("news_text", "").strip()

        if len(input_text) < 10:
            error = "Please enter at least 10 characters."
        else:
            cleaned = clean_text(input_text)
            X       = tfidf.transform([cleaned])

            y_pred = voting_model.predict(X)[0]
            proba  = voting_model.predict_proba(X)[0]

            fake_prob = round(proba[0] * 100, 1)
            real_prob = round(proba[1] * 100, 1)

            prediction = "REAL" if int(y_pred) == 1 else "FAKE"

            print(f"Input: {input_text[:60]}...")
            print(f"Prediction: {prediction} | FAKE: {fake_prob}% | REAL: {real_prob}%")

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        input_text=input_text,
        fake_prob=fake_prob,
        real_prob=real_prob
    )

if __name__ == "__main__":
    app.run(debug=True)