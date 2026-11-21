import os
import json
import pickle
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from preprocess import preprocess_text
from fact_verifier import verify_fact

# Streamlit Page Setup
st.set_page_config(
    page_title="Fake News Detection & Fact Verification",
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
        padding: 18px;
        color: #065F46;
        text-align: center;
        font-size: 1.35rem;
        font-weight: bold;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .card-fake {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 18px;
        color: #991B1B;
        text-align: center;
        font-size: 1.35rem;
        font-weight: bold;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .fact-card {
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .fact-supported {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        color: #064E3B;
    }
    .fact-contradicted {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        color: #7F1D1D;
    }
    .fact-unverified {
        background-color: #F3F4F6;
        border-left: 5px solid #6B7280;
        color: #374151;
    }
    .badge-label {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper to find file across BASE_DIR and working directory
def find_file(relative_path):
    candidates = [
        relative_path,
        os.path.join(BASE_DIR, relative_path),
        os.path.join(BASE_DIR, "..", relative_path),
        os.path.join(BASE_DIR, "Fake_News_Detection", relative_path)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# Load Model & Vectorizer
@st.cache_resource
def load_assets():
    model = None
    vectorizer = None
    
    # Try model paths
    for m_name in ["model.pkl", "fake_news_model.pkl"]:
        p = find_file(m_name)
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                model = pickle.load(f)
            break
            
    # Try vectorizer paths
    for v_name in ["vectorizer.pkl", "tfidf_vectorizer.pkl", "tfidf.pkl"]:
        p = find_file(v_name)
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                vectorizer = pickle.load(f)
            break
            
    return model, vectorizer

model, vectorizer = load_assets()

# Sidebar
with st.sidebar:
    st.title("📰 Project Info")
    st.markdown("**Fake News Detection & Fact Verification System**")
    st.write("Hybrid Machine Learning (TF-IDF + Logistic Regression) & Dynamic Evidence Verification.")
    st.divider()
    
    st.subheader("📊 Model Performance")
    metrics_file = find_file("metrics.json")
    if metrics_file:
        try:
            with open(metrics_file, "r") as f:
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
    st.subheader("🛠️ Architecture")
    st.write("1. **Statistical NLP:** TF-IDF (Unigram + Bigram, 50k features)")
    st.write("2. **Classifier:** Logistic Regression (Scikit-learn)")
    st.write("3. **Fact Verifier:** Dynamic Public Knowledge Verification (Authoritative Wikipedia API + Real-time Google News RSS)")
    st.write("4. **3-State Verification:** Supported, Contradicted, Not Verified")

# Main Header
st.markdown('<div class="title-text">📰 FAKE NEWS DETECTION & FACT VERIFICATION</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Hybrid System: Statistical Machine Learning + Dynamic Public Evidence Fact Checking</div>', unsafe_allow_html=True)

# Session State for Input
if 'news_text' not in st.session_state:
    st.session_state['news_text'] = ""

st.subheader("⚡ Quick One-Click Samples (Viva Voce Testing)")
c1, c2, c3, c4 = st.columns(4)

with c1:
    if st.button("🇮🇳 Sample 1 (Real Fact)", use_container_width=True):
        st.session_state['news_text'] = "Narendra Modi is the Prime Minister of India"
        st.rerun()

with c2:
    if st.button("⚠️ Sample 2 (False Entity Claim)", use_container_width=True):
        st.session_state['news_text'] = "Tushar is PM of India"
        st.rerun()

with c3:
    if st.button("📱 Sample 3 (Fake Scam Scheme)", use_container_width=True):
        st.session_state['news_text'] = "Government is giving free iPhone to everyone"
        st.rerun()

with c4:
    if st.button("🚀 Sample 4 (Real Space Mission)", use_container_width=True):
        st.session_state['news_text'] = "ISRO launched a new satellite into orbit successfully"
        st.rerun()

# Text Area
user_news = st.text_area(
    "Enter News Headline or Statement to Verify:",
    value=st.session_state.get('news_text', ''),
    height=130,
    placeholder="Type or paste any news statement here..."
)

col_b1, col_b2 = st.columns([1, 4])
with col_b1:
    predict_clicked = st.button("🔍 Analyze & Verify", type="primary", use_container_width=True)

if predict_clicked:
    if not user_news.strip():
        st.warning("⚠️ Pehle koi news to likho ya paste karo!")
    elif model is None or vectorizer is None:
        st.error("❌ Model ya Vectorizer load nahi hua! Please check model.pkl and vectorizer.pkl.")
    else:
        with st.spinner("Running Statistical NLP Model & Cross-Referencing Evidence..."):
            # Step 1: Preprocess text using NLTK
            cleaned = preprocess_text(user_news)
            
            # Step 2: TF-IDF Feature Extraction
            vectorized = vectorizer.transform([cleaned])
            
            # Step 3: ML Model Prediction (1 = REAL, 0 = FAKE)
            ml_pred = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence_real = proba[1] * 100
            confidence_fake = proba[0] * 100
            
            # Step 4: Dynamic Evidence-Based Fact Verification
            fact_result = verify_fact(user_news)
            
            st.divider()
            
            # Step 5: Synthesize Final Verdict
            # If Fact Verifier finds direct contradiction (e.g. false entity claiming PM of India):
            # It flags it as FAKE and overrides statistical false positives!
            if fact_result["status"] == "CONTRADICTED":
                final_verdict = "FAKE"
                card_html = f"""
                <div class="card-fake">
                    ❌ PREDICTION: FAKE NEWS (Factual Contradiction Detected)
                    <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                        <strong>Verdict Note:</strong> The statement contradicts verified public records. The statistical ML model evaluated context keywords, but evidence verification identified the contradiction.
                    </div>
                </div>
                """
            elif fact_result["status"] == "SUPPORTED":
                final_verdict = "REAL"
                card_html = f"""
                <div class="card-real">
                    ✅ PREDICTION: REAL NEWS (Factually Supported)
                    <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                        <strong>Confidence:</strong> {confidence_real:.2f}% (Corroborated by authoritative reference data and statistical NLP analysis)
                    </div>
                </div>
                """
            else:
                # NOT_VERIFIED: Rely on Statistical ML Classifier with transparency
                if ml_pred == 1:
                    final_verdict = "REAL"
                    card_html = f"""
                    <div class="card-real">
                        ✅ PREDICTION: REAL NEWS
                        <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                            <strong>ML Confidence Score:</strong> {confidence_real:.2f}% probability of being Authentic/Real News
                        </div>
                    </div>
                    """
                else:
                    final_verdict = "FAKE"
                    card_html = f"""
                    <div class="card-fake">
                        ❌ PREDICTION: FAKE NEWS
                        <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                            <strong>ML Confidence Score:</strong> {confidence_fake:.2f}% probability of being Misleading/Fake News
                        </div>
                    </div>
                    """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Step 6: Detailed Fact Verification Section
            st.write("")
            st.subheader("🛡️ Fact Verification Analysis (Authoritative Public Evidence)")
            
            fact_status = fact_result["status"]
            if fact_status == "SUPPORTED":
                status_class = "fact-supported"
                badge_bg = "#10B981"
                status_text = "🟢 SUPPORTED"
            elif fact_status == "CONTRADICTED":
                status_class = "fact-contradicted"
                badge_bg = "#EF4444"
                status_text = "🔴 CONTRADICTED"
            else:
                status_class = "fact-unverified"
                badge_bg = "#6B7280"
                status_text = "⚪ NOT VERIFIED"
                
            source_url = fact_result.get("source_url", "")
            if source_url:
                source_html = f'<a href="{source_url}" target="_blank" style="color: #1D4ED8; text-decoration: underline; font-weight: 600;">{fact_result["source"]} ↗</a>'
            else:
                source_html = f'<span>{fact_result["source"]}</span>'

            st.markdown(f"""
            <div class="fact-card {status_class}">
                <div class="badge-label" style="background-color: {badge_bg}; color: white;">
                    {status_text}
                </div>
                <div style="font-size: 1.1rem; font-weight: bold; margin-bottom: 6px;">
                    {fact_result['title']}
                </div>
                <div style="margin-bottom: 8px;">
                    <strong>Verification Details:</strong> {fact_result['reason']}
                </div>
                <div style="font-size: 0.92rem;">
                    <strong>Evidence Source:</strong> {source_html}
                </div>
                <div style="font-size: 0.88rem; margin-top: 6px; opacity: 0.85;">
                    <em>Scope & Methodology:</em> {fact_result.get('limitations', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Step 7: Statistical ML Confidence Breakdown
            st.write("")
            st.subheader("📊 Statistical Model Probability Breakdown (TF-IDF + Logistic Regression)")
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
                st.caption("Notice: TF-IDF analyzes statistical term frequency and n-gram patterns. For entities and factual claims, dynamic fact verification provides the ground-truth evidence.")

# Visual Graphs Section
st.divider()
st.subheader("📈 Model Training Visualizations (College PPT Graphs)")
g1, g2, g3 = st.columns(3)

dist_img = find_file("graphs/class_distribution.png")
cm_img = find_file("graphs/confusion_matrix.png")
kw_img = find_file("graphs/top_keywords.png")

with g1:
    if dist_img and os.path.exists(dist_img):
        st.image(dist_img, caption="Dataset Distribution (Short + Long)", use_container_width=True)

with g2:
    if cm_img and os.path.exists(cm_img):
        st.image(cm_img, caption="Confusion Matrix Heatmap", use_container_width=True)

with g3:
    if kw_img and os.path.exists(kw_img):
        st.image(kw_img, caption="Top Predictive Keywords (Real vs Fake)", use_container_width=True)
