"""
Streamlit UI — Resume Role Classifier
Run: streamlit run streamlit_app.py
"""
from dotenv import load_dotenv
import streamlit as st
import requests
import os
load_dotenv()
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Role Classifier",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("API_URL")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: #a0aec0;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #3d5a80;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    text-align: center;
    box-shadow: 0 8px 32px rgba(61, 90, 128, 0.3);
}
.role-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #63b3ed;
    letter-spacing: -0.5px;
}
.confidence-badge {
    display: inline-block;
    background: linear-gradient(90deg, #38a169, #48bb78);
    color: white;
    font-weight: 700;
    font-size: 1.4rem;
    padding: 0.4rem 1.4rem;
    border-radius: 50px;
    margin-top: 0.8rem;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(56, 161, 105, 0.4);
}

/* Top predictions */
.pred-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    border-left: 3px solid #3d5a80;
}
.pred-role { color: #e2e8f0; font-weight: 500; }
.pred-pct  { color: #63b3ed; font-weight: 700; font-size: 0.95rem; }

/* Upload zone */
.upload-hint {
    color: #718096;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎯 Resume Role Classifier</h1>
    <p>Upload your resume — our ML model predicts your job role category with confidence score</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: API status ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL)

    st.subheader("API Status")
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success(" API Online")
            st.write(
                f"Model loaded: { '✅' if data.get('model_loaded') else '❌' }")
            st.write(f"Version: {data.get('version', 'N/A')}")
        else:
            st.error("❌ API Error")
    except Exception:
        st.error("❌ API Offline")
        st.caption(f"Make sure Flask is running at {api_url}")

    st.subheader("Supported Formats")
    st.markdown("- 📄 PDF\n- 📝 DOCX\n- 📃 TXT")
    st.caption(f"Max size: 10 MB")


# ── File upload ────────────────────────────────────────────────────────────────
st.subheader("📤 Upload Your Resume")
uploaded_file = st.file_uploader(
    "Choose a resume file",
    type=["pdf", "docx", "txt"],
    help="Upload a PDF, Word document, or plain text resume",
)
st.markdown('<p class="upload-hint">Supported: .pdf · .docx · .txt &nbsp;|&nbsp; Max 10 MB</p>',
            unsafe_allow_html=True)


# ── Predict ────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button(
            "🔍 Predict Role", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("Analyzing resume..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(
                    f"{api_url}/predict", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    role = result["predicted_role"]
                    conf = result["confidence"]
                    top = result["top_predictions"]

                    # ── Main result card ──
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="color:#a0aec0; font-size:0.9rem; margin-bottom:0.5rem;">PREDICTED ROLE</div>
                        <div class="role-label">{role}</div>
                        <div style="margin-top:1rem;">
                            <span class="confidence-badge">✓ {conf:.1f}% Confidence</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Confidence bar ──
                    st.progress(conf / 100)

                    # ── Top predictions ──
                    st.subheader("📊 Top Predictions")
                    for pred in top:
                        bar_width = pred["confidence"]
                        st.markdown(f"""
                        <div class="pred-row">
                            <span class="pred-role">{pred['role']}</span>
                            <span class="pred-pct">{pred['confidence']:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(pred["confidence"] / 100)

                    # ── File info ──
                    with st.expander("📁 File Info"):
                        st.write(f"**Filename:** {uploaded_file.name}")
                        st.write(
                            f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")

                elif response.status_code == 503:
                    st.error(
                        "⚠️ Model not loaded. Run `python src/train.py` to train the model first.")
                else:
                    err = response.json()
                    st.error(
                        f" Error {response.status_code}: {err.get('message', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f" Cannot connect to API at `{api_url}`.\n\nStart the Flask server: `python app/app.py`")
            except Exception as e:
                st.error(f" Unexpected error: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Resume Role Classifier"
    "</p>",
    unsafe_allow_html=True,
)
