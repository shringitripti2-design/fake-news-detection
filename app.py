import streamlit as st
import pickle
import re

st.set_page_config(page_title="Fake News Detection", page_icon="📰")
st.title("📰 Fake News Detection")
st.write("News likho aur check karo ki ye REAL hai ya FAKE hai.")

# Model aur Vectorizer load karna
try:
    model = pickle.load(open("fake_news_model.pkl", "rb"))
except:
    model = pickle.load(open("model.pkl", "rb"))

try:
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
except:
    try:
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    except:
        vectorizer = pickle.load(open("tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

news_text = st.text_area("Yahan News likho:", height=150)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Pehle koi news to likho!")
    else:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success(f"✅ REAL NEWS hai! Confidence: {max(proba)*100:.2f}%")
        else:
            st.error(f"❌ FAKE NEWS hai! Confidence: {max(proba)*100:.2f}%")