# FakeNewsPrediction 🔍

A machine learning-based fake news detection system using a hybrid ensemble methodology combining TF-IDF feature extraction with a Voting Classifier (Logistic Regression + Decision Tree + Random Forest).

## 📊 Datasets

Download the datasets and place them in the `data/` folder:

- **ISOT Fake News Dataset** (Fake.csv, True.csv): https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **WELFake Dataset** (WELFake_Dataset.csv): https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

## 🚀 How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python src/train_new.py`
3. Run the web app: `python app.py`
4. Open browser at `http://localhost:5000`

## 🛠️ Tech Stack

- Python, Flask, Scikit-learn, Pandas, NumPy
- TF-IDF Vectorization
- Ensemble Voting Classifier (94% accuracy)