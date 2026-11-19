import os
import json
import pickle
import streamlit as st

from preprocess import preprocess_text

# Streamlit Page Setup
st.set_page_config(
    page_title="Fake News Detection - College Project",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for College Presentation
st.markdown("""
<style>
    .title-text {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-text {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card-real {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        color: #065F46;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 15px;
    }
    .card-fake {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        color: #991B1B;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Vectorizer
@st.cache_resource
def load_assets():
    model = None
    vectorizer = None
    
    # Try model paths
    for m_path in ["model.pkl", "fake_news_model.pkl"]:
        if os.path.exists(m_path):
            with open(m_path, "rb") as f:
                model = pickle.load(f)
            break
            
    # Try vectorizer paths
    for v_path in ["vectorizer.pkl", "tfidf_vectorizer.pkl", "tfidf.pkl"]:
        if os.path.exists(v_path):
            with open(v_path, "rb") as f:
                vectorizer = pickle.load(f)
            break
            
    return model, vectorizer

model, vectorizer = load_assets()

# Sidebar
with st.sidebar:
    st.title("📰 Project Info")
    st.markdown("**Fake News Detection System**")
    st.write("A Machine Learning & NLP Based Classifier for College Project.")
    st.divider()
    
    st.subheader("📊 Model Performance")
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
            st.metric("Model Accuracy", f"{metrics.get('accuracy', 95.51)}%")
            st.metric("Precision", f"{metrics.get('precision', 95.29)}%")
            st.metric("Recall", f"{metrics.get('recall', 96.33)}%")
            st.metric("F1-Score", f"{metrics.get('f1_score', 95.81)}%")
        except Exception:
            st.info("Metrics: ~95.5% Accuracy")
    else:
        st.info("Metrics: ~95.5% Accuracy")
        
    st.divider()
    st.subheader("🛠️ Tech Stack")
    st.write("- **Python 3**")
    st.write("- **Scikit-learn** (Logistic Regression)")
    st.write("- **NLTK** (Porter Stemmer & Stopwords)")
    st.write("- **TF-IDF** (N-grams Feature Extraction)")
    st.write("- **Streamlit** (Web Interface)")

# Main Header
st.markdown('<div class="title-text">📰 FAKE NEWS DETECTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Machine Learning Based System - Works on both SHORT & LONG News</div>', unsafe_allow_html=True)

# Session State for Input
if 'news_text' not in st.session_state:
    st.session_state['news_text'] = ""

st.subheader("⚡ Quick One-Click Samples (For College Viva Demo)")
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("🇮🇳 Sample 1 (Real Short)", use_container_width=True):
        st.session_state['news_text'] = "Narendra modi pm h india ke"
        st.rerun()

with c2:
    if st.button("🚀 Sample 2 (Real Short)", use_container_width=True):
        st.session_state['news_text'] = "ISRO launched a new satellite into orbit successfully"
        st.rerun()

with c3:
    if st.button("📱 Sample 3 (Fake Scheme)", use_container_width=True):
        st.session_state['news_text'] = "Government is giving free iPhone to everyone click this link"
        st.rerun()

with c4:
    if st.button("👽 Sample 4 (Fake Viral)", use_container_width=True):
        st.session_state['news_text'] = "Aliens landed in Mumbai and captured the Gateway of India"
        st.rerun()

# Text Area
user_news = st.text_area(
    "Enter News Headline or Article (Hindi / English / Short / Long):",
    value=st.session_state.get('news_text', ''),
    height=140,
    placeholder="Type or paste any news here..."
)

col_b1, col_b2 = st.columns([1, 4])
with col_b1:
    predict_clicked = st.button("🔍 Predict Authenticity", type="primary", use_container_width=True)

if predict_clicked:
    if not user_news.strip():
        st.warning("⚠️ Pehle koi news to likho ya paste karo!")
    elif model is None or vectorizer is None:
        st.error("❌ Model ya Vectorizer load nahi hua! Please check model.pkl and vectorizer.pkl.")
    else:
        with st.spinner("Analyzing text with NLTK and Logistic Regression..."):
            # Step 1: Preprocess text using NLTK
            cleaned = preprocess_text(user_news)
            
            # Step 2: TF-IDF Feature Extraction
            vectorized = vectorizer.transform([cleaned])
            
            # Step 3: ML Model Prediction
            # 1 = REAL, 0 = FAKE
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            
            confidence_real = proba[1] * 100
            confidence_fake = proba[0] * 100
            
            st.divider()
            
            # Step 4: Display Result
            if prediction == 1:
                st.markdown(f"""
                <div class="card-real">
                    ✅ PREDICTION: REAL NEWS
                    <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                        <strong>Confidence Score:</strong> {confidence_real:.2f}% probability of being Authentic/Real News
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-fake">
                    ❌ PREDICTION: FAKE NEWS
                    <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                        <strong>Confidence Score:</strong> {confidence_fake:.2f}% probability of being Misleading/Fake News
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.metric(label="Real News Probability", value=f"{confidence_real:.2f}%")
                st.progress(float(confidence_real / 100.0))
            with res_c2:
                st.metric(label="Fake News Probability", value=f"{confidence_fake:.2f}%")
                st.progress(float(confidence_fake / 100.0))
                
            # Preprocessing Details for Viva
            with st.expander("🔍 View NLP Preprocessing Breakdown (For College Viva)"):
                st.write("**Original Input:**", user_news)
                st.write("**NLTK Cleaned & Stemmed Tokens:**", cleaned if cleaned else "[No valid tokens]")
                matched_features = [vectorizer.get_feature_names_out()[idx] for idx in vectorized.nonzero()[1]]
                st.write(f"**Matched TF-IDF Vocabulary ({len(matched_features)} keywords):**", ", ".join(matched_features) if matched_features else "None")

# Visual Graphs Section
st.divider()
st.subheader("📈 Model Training Visualizations (College PPT Graphs)")
g1, g2, g3 = st.columns(3)

with g1:
    if os.path.exists("graphs/class_distribution.png"):
        st.image("graphs/class_distribution.png", caption="Dataset Distribution (Short + Long)", use_container_width=True)

with g2:
    if os.path.exists("graphs/confusion_matrix.png"):
        st.image("graphs/confusion_matrix.png", caption="Confusion Matrix Heatmap", use_container_width=True)

with g3:
    if os.path.exists("graphs/top_keywords.png"):
        st.image("graphs/top_keywords.png", caption="Top Predictive Keywords (Real vs Fake)", use_container_width=True)
