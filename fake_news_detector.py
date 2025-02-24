import pandas as pd
import numpy as np
import re
import joblib
import streamlit as st
import nltk
import gdown
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- FIX: Manually Check and Download Stopwords Only If Needed ---
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# --- NEW: Download Dataset from Google Drive ---
true_news_url = "https://drive.google.com/uc?id=1MUyqmryTEEsVt0NF61wGvVxXo66sNJY4"  # True.csv
fake_news_url = "https://drive.google.com/uc?id=1e1x7vUjDBqquC-lYjz6-BYDTCvpFViNe"  # Fake.csv

true_news_path = "True.csv"
fake_news_path = "Fake.csv"

gdown.download(true_news_url, true_news_path, quiet=False)
gdown.download(fake_news_url, fake_news_path, quiet=False)

# --- LOAD DATASET ---
df_fake = pd.read_csv(fake_news_path)
df_real = pd.read_csv(true_news_path)

# Assign labels
df_fake["label"] = 1  # Fake news
df_real["label"] = 0  # Real news

# Combine datasets
df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

# --- TEXT PREPROCESSING FUNCTION ---
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply text preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# --- CONVERT TEXT TO FEATURES ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"]).toarray()
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# --- STREAMLIT APP ---
st.title("📰 Fake News Detector")

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# User input
user_input = st.text_area("Enter news text:")

if st.button("Check"):
    if user_input.strip() != "":
        # Preprocess and predict
        user_text_vectorized = vectorizer.transform([preprocess_text(user_input)])
        prediction = model.predict(user_text_vectorized)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 This is likely FAKE news!")
        else:
            st.success("✅ This is likely REAL news!")
    else:
        st.warning("Please enter some text.")
