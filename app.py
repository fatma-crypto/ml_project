import streamlit as st
from page_modules import home
from page_modules import upload
from page_modules import visualization
from page_modules import preprocessing
from page_modules import model_selection
from page_modules import evaluation
from page_modules import predict

st.set_page_config(
    page_title="Project Risk Assessment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2128;
    --border:        #30363d;
    --accent:        #58a6ff;
    --accent-green:  #3fb950;
    --accent-red:    #f85149;
    --accent-orange: #d29922;
    --text-primary:  #e6edf3;
    --text-muted:    #8b949e;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Navigation buttons */
div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted) !important;
    padding: 10px 14px;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s;
    margin-bottom: 2px;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(88,166,255,0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* All primary buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Cards */
.risk-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Metric overrides */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
div[data-testid="metric-container"] label { color: var(--text-muted) !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; }

/* Selectbox / inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stTextInput > div > input,
.stNumberInput > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* DataFrames */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-secondary) !important; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* Expander */
.streamlit-expanderHeader { background: var(--bg-card) !important; border-radius: 8px !important; }

/* Headers */
h1 { font-size: 28px !important; font-weight: 700 !important; color: var(--text-primary) !important; }
h2 { font-size: 20px !important; font-weight: 600 !important; color: var(--text-primary) !important; }
h3 { font-size: 16px !important; font-weight: 500 !important; color: var(--text-muted) !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* Success / Error / Info / Warning */
.stSuccess, .stError, .stInfo, .stWarning { border-radius: 8px !important; }

/* Slider */
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }

/* Upload area */
.stFileUploader { background: var(--bg-card) !important; border: 2px dashed var(--border) !important; border-radius: 12px !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Radio buttons */
.stRadio > div { gap: 8px !important; }
.stRadio label { color: var(--text-primary) !important; }

/* Checkboxes */
.stCheckbox label { color: var(--text-primary) !important; }

/* Mono badge */
.badge {
    display: inline-block;
    background: rgba(88,166,255,0.15);
    color: var(--accent);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ──────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None

# ── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Risk Assessment")
    st.markdown("---")

    pages = [
        ("🏠 Home",             "Home"),
        ("📁 1 · File Upload",  "Data pipeline starts here"),
        ("📊 2 · Visualization","Explore your data"),
        ("⚙️ 3 · Preprocessing","Clean & transform"),
        ("🤖 4 · Model",        "Train a classifier"),
        ("📈 5 · Evaluation",   "Review results"),
        ("🔮 Predict",          "Run predictions"),
    ]

    for label, tip in pages:
        if st.button(label, key=f"nav_{label}", help=tip):
            st.session_state.page = label
            st.rerun()

    st.markdown("---")
    if st.session_state.uploaded_df is not None:
        n = len(st.session_state.uploaded_df)
        st.markdown(f"<span class='badge'>✓ {n} rows loaded</span>", unsafe_allow_html=True)
    if st.session_state.trained_model is not None:
        st.markdown("<span class='badge'>✓ Model ready</span>", unsafe_allow_html=True)

# ── Page Router ─────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "🏠 Home":
    from page_modules import home
    home.show()
elif page == "📁 1 · File Upload":
    from page_modules import upload
    upload.show()
elif page == "📊 2 · Visualization":
    from page_modules import visualization
    visualization.show()
elif page == "⚙️ 3 · Preprocessing":
    from page_modules import preprocessing
    preprocessing.show()
elif page == "🤖 4 · Model":
    from page_modules import model_selection
    model_selection.show()
elif page == "📈 5 · Evaluation":
    from page_modules import evaluation
    evaluation.show()
elif page == "🔮 Predict":
    from page_modules import predict
    predict.show()
