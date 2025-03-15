import pandas as pd
import numpy as np
import re
import joblib
import streamlit as st
import nltk
import requests
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load stopwords
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

# --- LOAD AND TRAIN MODEL ---
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

true_news["label"] = 0  # 0 for real news
fake_news["label"] = 1  # 1 for fake news

news_data = pd.concat([true_news, fake_news])
news_data = news_data.sample(frac=1).reset_index(drop=True)

# --- TEXT PREPROCESSING FUNCTION ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

news_data["clean_text"] = news_data["text"].apply(preprocess_text)
X = news_data["clean_text"]
y = news_data["label"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# --- STREAMLIT UI ---
st.set_page_config(page_title="Fake News Detector", layout="wide", initial_sidebar_state="expanded")
st.title("📰 Fake News Detector")

# --- USER INPUT ---
user_input = st.text_area("Enter news text:")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a CSV or TXT file for bulk analysis", type=["csv", "txt"])

# --- CHECK BUTTON ---
if st.button("Check"):
    if user_input.strip():
        user_text_vectorized = vectorizer.transform([preprocess_text(user_input)])
        prediction = model.predict(user_text_vectorized)[0]
        confidence = model.predict_proba(user_text_vectorized)[0][prediction]

        if prediction == 1:
            st.error(f"🚨 This is likely FAKE news! (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ This is likely REAL news! (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text.")

# --- BULK FILE PROCESSING ---
if uploaded_file:
    st.subheader("📂 File Processing Results")
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame({"text": uploaded_file.read().decode("utf-8").split("\n")})
    
    df = df.dropna()
    df["clean_text"] = df["text"].apply(preprocess_text)
    df_vectorized = vectorizer.transform(df["clean_text"])
    df["Prediction"] = model.predict(df_vectorized)
    df["Confidence"] = model.predict_proba(df_vectorized).max(axis=1)
    
    df["Prediction"] = df["Prediction"].map({1: "FAKE", 0: "REAL"})
    st.dataframe(df)

# --- RECENT NEWS VERIFICATION ---
def get_latest_news():
    api_key = "aca9e60a63514cbbba1197a63e0b507a"  # Replace with your actual API key
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url).json()
    return response["articles"] if "articles" in response else []

if st.button("🔍 Check Latest News"):
    articles = get_latest_news()
    if articles:
        st.subheader("📢 Recent News Headlines")
        for article in articles[:5]:
            st.markdown(f"**{article['title']}**")
            st.write(article["description"])
            user_text_vectorized = vectorizer.transform([preprocess_text(article["title"])] )
            prediction = model.predict(user_text_vectorized)[0]
            confidence = model.predict_proba(user_text_vectorized)[0][prediction]

            if prediction == 1:
                st.error(f"🚨 FAKE News Detected! (Confidence: {confidence:.2f})")
            else:
                st.success(f"✅ REAL News Detected! (Confidence: {confidence:.2f})")
    else:
        st.warning("Could not fetch news articles. Please try again later.")
