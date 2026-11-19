import os
import json
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

from preprocess import preprocess_text

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-card-real {
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
    .result-card-fake {
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
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Cache Model and Vectorizer Loading
@st.cache_resource
def load_ml_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading trained model: {e}")
        return None, None

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

model, vectorizer = load_ml_assets()
metrics = load_metrics()

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=90)
    st.title("Navigation")
    menu = st.radio(
        "Go to Section:",
        [
            "🏠 Home / Predictor",
            "📊 Model Evaluation",
            "📈 Visual Analysis (Graphs)",
            "📁 Dataset Information",
            "🎓 About Project & Viva"
        ]
    )
    st.divider()
    
    st.subheader("⚙️ System Status")
    if model is not None and vectorizer is not None:
        st.success("✅ Model: Loaded (Logistic Regression)")
        st.success("✅ Vectorizer: Loaded (TF-IDF)")
    else:
        st.error("⚠️ Model not loaded! Please run `python train_model.py` first.")
        
    st.caption("College Project: Fake News Detection System")

# ==========================================
# PAGE 1: HOME / PREDICTOR
# ==========================================
if menu == "🏠 Home / Predictor":
    st.markdown('<div class="main-title">FAKE NEWS DETECTION SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Machine Learning & NLP Based Web Application for College Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This system utilizes **Natural Language Processing (NLTK)** and a **Scikit-learn Logistic Regression** classifier 
    trained on genuine labeled news articles to determine the credibility and authenticity of any given news content.
    """)
    
    st.divider()

    # Pre-defined College Demonstration Samples
    st.subheader("📝 Try Sample News Articles (One-Click Viva Demo)")
    sample_col1, sample_col2, sample_col3 = st.columns(3)

    sample_real = (
        "WASHINGTON (Reuters) - The Federal Reserve kept interest rates unchanged on Wednesday "
        "and signaled that borrowing costs are likely to remain on hold for the foreseeable future, "
        "as moderate economic growth and strong job creation are expected to continue."
    )
    sample_fake = (
        "BREAKING EXCLUSIVE: Secret documents leak reveals aliens have taken control of world governments "
        "and will replace all paper money with galactic space tokens by midnight! Share this before it gets banned!"
    )

    if 'news_input' not in st.session_state:
        st.session_state['news_input'] = ""

    with sample_col1:
        if st.button("📌 Load Sample REAL News", use_container_width=True):
            st.session_state['news_input'] = sample_real
            st.rerun()

    with sample_col2:
        if st.button("🚨 Load Sample FAKE News", use_container_width=True):
            st.session_state['news_input'] = sample_fake
            st.rerun()

    with sample_col3:
        if st.button("🧹 Clear Input Box", use_container_width=True):
            st.session_state['news_input'] = ""
            st.rerun()

    st.subheader("📰 Enter News Article for Verification")
    user_input = st.text_area(
        label="Paste article headline or full body content below:",
        value=st.session_state.get('news_input', ''),
        height=180,
        placeholder="Type or paste the news article text here..."
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        predict_clicked = st.button("🔍 Predict Authenticity", type="primary", use_container_width=True)

    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter or paste some text before clicking Predict!")
        elif model is None or vectorizer is None:
            st.error("❌ Model or Vectorizer is not loaded! Please run `python train_model.py` to generate the model files.")
        else:
            with st.spinner("Analyzing text with NLTK and Logistic Regression..."):
                # STEP 2: NLP Analysis (Preprocessing)
                cleaned_text = preprocess_text(user_input)

                if not cleaned_text.strip():
                    st.warning("⚠️ The input text contains only symbols or stopwords. Please provide informative news sentences.")
                else:
                    # STEP 3: Machine Learning Prediction
                    input_tfidf = vectorizer.transform([cleaned_text])
                    prediction = model.predict(input_tfidf)[0]
                    prediction_proba = model.predict_proba(input_tfidf)[0]

                    # 1 = FAKE, 0 = REAL
                    confidence_fake = prediction_proba[1] * 100
                    confidence_real = prediction_proba[0] * 100

                    # STEP 4: User Result Display
                    st.subheader("🎯 Classification Result")
                    if prediction == 1:
                        # FAKE Result in RED
                        st.markdown(f"""
                        <div class="result-card-fake">
                            🚨 PREDICTION: FAKE NEWS
                            <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                                <strong>Confidence Score:</strong> {confidence_fake:.2f}% probability of being Fake News
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # REAL Result in GREEN
                        st.markdown(f"""
                        <div class="result-card-real">
                            ✅ PREDICTION: REAL NEWS
                            <div style="font-size: 1.05rem; font-weight: normal; margin-top: 8px;">
                                <strong>Confidence Score:</strong> {confidence_real:.2f}% probability of being Authentic/Real News
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.write("")
                    # Confidence breakdown progress bars
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric(label="Real News Probability", value=f"{confidence_real:.2f}%")
                        st.progress(float(confidence_real / 100.0))
                    with res_col2:
                        st.metric(label="Fake News Probability", value=f"{confidence_fake:.2f}%")
                        st.progress(float(confidence_fake / 100.0))

                    # NLP Preprocessing Inspection expander
                    with st.expander("🔍 View NLP Preprocessing Breakdown (For Viva Demo)"):
                        st.write("**1. Raw Input Text:**")
                        st.code(user_input, language="text")
                        st.write("**2. After NLTK Cleaning & Stemming:**")
                        st.code(cleaned_text, language="text")
                        st.write("**3. Extracted TF-IDF Features:**")
                        non_zero_indices = input_tfidf.nonzero()[1]
                        words_found = [vectorizer.get_feature_names_out()[idx] for idx in non_zero_indices]
                        st.write(f"Matched {len(words_found)} vocabulary keywords:", ", ".join(words_found[:30]) if words_found else "None")

# ==========================================
# PAGE 2: MODEL EVALUATION
# ==========================================
elif menu == "📊 Model Evaluation":
    st.title("📊 Model Evaluation & Performance Metrics")
    st.write("Quantitative evaluation of the Scikit-learn Logistic Regression classifier on the testing set (20% hold-out data).")
    
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Model Accuracy", f"{metrics['accuracy']}%")
        m2.metric("🔍 Precision", f"{metrics['precision']}%")
        m3.metric("📈 Recall", f"{metrics['recall']}%")
        m4.metric("⚖️ F1-Score", f"{metrics['f1_score']}%")
        
        st.divider()

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("📋 Split Summary")
            st.write(f"- **Total Dataset Samples:** {metrics.get('total_samples', 'N/A')}")
            st.write(f"- **Training Samples (80%):** {metrics.get('train_samples', 'N/A')}")
            st.write(f"- **Testing Samples (20%):** {metrics.get('test_samples', 'N/A')}")
            st.write(f"- **Real Articles:** {metrics.get('real_count', 'N/A')}")
            st.write(f"- **Fake Articles:** {metrics.get('fake_count', 'N/A')}")

        with col_m2:
            st.subheader("🔢 Confusion Matrix Counts")
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            st.write(f"- **True Negatives (Correctly Real):** {cm[0][0]}")
            st.write(f"- **False Positives (Real flagged Fake):** {cm[0][1]}")
            st.write(f"- **False Negatives (Fake flagged Real):** {cm[1][0]}")
            st.write(f"- **True Positives (Correctly Fake):** {cm[1][1]}")
    else:
        st.info("ℹ️ Evaluation metrics will appear once you train the model via `python train_model.py`.")

# ==========================================
# PAGE 3: VISUAL ANALYSIS (GRAPHS)
# ==========================================
elif menu == "📈 Visual Analysis (Graphs)":
    st.title("📈 Matplotlib Data Visualizations")
    st.write("Visual graphs generated during model training for college presentation and project viva.")

    cm_path = os.path.join(GRAPHS_DIR, "confusion_matrix.png")
    dist_path = os.path.join(GRAPHS_DIR, "class_distribution.png")
    keywords_path = os.path.join(GRAPHS_DIR, "top_keywords.png")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.subheader("1. Confusion Matrix Heatmap")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix of Logistic Regression on Test Set", use_container_width=True)
        else:
            st.warning("Confusion matrix graph not found. Run `python train_model.py`.")

    with g_col2:
        st.subheader("2. Dataset Class Distribution")
        if os.path.exists(dist_path):
            st.image(dist_path, caption="Distribution of Real vs Fake News in Dataset", use_container_width=True)
        else:
            st.warning("Distribution graph not found. Run `python train_model.py`.")

    st.divider()
    st.subheader("3. Top Predictive Keywords (Feature Importance)")
    if os.path.exists(keywords_path):
        st.image(keywords_path, caption="Most Influential Words for Real vs Fake Classification", use_container_width=True)
    else:
        st.warning("Keywords graph not found. Run `python train_model.py`.")

# ==========================================
# PAGE 4: DATASET INFORMATION
# ==========================================
elif menu == "📁 Dataset Information":
    st.title("📁 Dataset Information")
    st.write("Overview of the genuine labeled dataset used for training and testing.")

    csv_candidates = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.csv')] if os.path.exists(DATASET_DIR) else []
    
    if csv_candidates:
        dataset_file = csv_candidates[0]
        st.success(f"✅ Active Dataset: `{os.path.basename(dataset_file)}` ({os.path.getsize(dataset_file):,} bytes)")
        try:
            df_preview = pd.read_csv(dataset_file, nrows=10)
            st.subheader("🔍 Dataset Preview (First 10 Rows)")
            st.dataframe(df_preview, use_container_width=True)
            
            st.subheader("📊 Dataset Metadata")
            st.markdown("""
            - **Source:** Kaggle Fake and Real News Dataset (McIntire / George McIntire)
            - **Attributes:** `title`, `text`, `label`
            - **Target Labels:** `REAL` (Authentic News), `FAKE` (Misleading/False News)
            - **Preprocessing:** Title and body text concatenated, lowercased, stripped of URLs, symbols, stopwords, and stemmed.
            """)
        except Exception as e:
            st.error(f"Error reading dataset preview: {e}")
    else:
        st.info("ℹ️ Dataset file will be downloaded automatically when running `python train_model.py`.")

# ==========================================
# PAGE 5: ABOUT PROJECT & VIVA
# ==========================================
elif menu == "🎓 About Project & Viva":
    st.title("🎓 Project Synopsis & Viva Guide")
    st.markdown("""
    ### 📌 Title: Fake News Detection – A Machine Learning Based System
    
    #### 1. Objective:
    To build an automated machine learning software pipeline capable of classifying online news articles as Real or Fake based on linguistic analysis and statistical NLP feature extraction.
    
    #### 2. System Architecture & Workflow:
    1. **Article Input:** User pastes the news article into the web interface.
    2. **NLP Analysis:** NLTK cleans HTML, URLs, punctuation, removes stopwords, and performs Porter Stemming.
    3. **TF-IDF Extraction:** Scikit-learn transforms cleaned text into numerical word importance vectors.
    4. **Classification:** Scikit-learn Logistic Regression computes prediction and probability confidence scores.
    5. **User Result:** Displays Green (Real News) or Red (Fake News) badge with confidence percentages.
    
    #### 3. Technologies Used:
    - **Language:** Python
    - **ML Library:** Scikit-learn
    - **NLP:** NLTK
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Matplotlib
    - **Model Persistence:** Joblib
    - **Interface:** Streamlit
    - **IDE:** VS Code
    - **Source:** Kaggle
    
    ---
    ### 💡 Viva Voce Quick Revision Q&A:
    
    **Q1: Why use TF-IDF instead of simple Bag-of-Words?**  
    *A:* Bag-of-Words only counts word frequencies, giving high weight to words that appear often even if uninformative. TF-IDF balances frequency with inverse document frequency, giving high scores only to words that distinguish specific classes.
    
    **Q2: Why is Logistic Regression suitable for text classification?**  
    *A:* Logistic Regression handles high-dimensional, sparse TF-IDF vectors well, avoids overfitting via regularization, is computationally lightweight, and provides calibrated probability scores.
    
    **Q3: What role does NLTK play?**  
    *A:* NLTK provides stopword filters (removing words like 'the', 'is', 'at') and the Porter Stemmer (collapsing variations like 'connecting', 'connected' into root 'connect'), reducing vocabulary noise.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Fake News Detection System | College Project")
