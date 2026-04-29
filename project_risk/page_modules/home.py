import streamlit as st
import os


def show():
    st.markdown("# 🎯 Project Risk Assessment")
    st.markdown("#### AI-powered pipeline for predicting project risk levels")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='risk-card'>
            <h2>📁 Upload Data</h2>
            <p style='color:#8b949e;font-size:14px;'>Load your CSV or Excel dataset to begin the analysis pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='risk-card'>
            <h2>⚙️ Preprocess</h2>
            <p style='color:#8b949e;font-size:14px;'>Encode, normalize, handle missing values, outliers, and imbalance.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='risk-card'>
            <h2>🤖 Train & Predict</h2>
            <p style='color:#8b949e;font-size:14px;'>Select a model, train it, evaluate performance, and predict risk.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Pre-trained model status
    model_exists = os.path.exists("best_model.joblib")
    if model_exists:
        st.success("✅ Pre-trained model detected (`best_model.joblib`). You can jump straight to **🔮 Predict**.")
    else:
        st.info("ℹ️ No pre-trained model found. Use **📁 File Upload → ⚙️ Preprocessing → 🤖 Model** to train one.")

    st.markdown("### 📋 Pipeline Steps")
    steps = [
        ("1", "📁 File Upload",     "Upload CSV or Excel — supports multiple formats with validation."),
        ("2", "📊 Visualization",   "Explore distributions, correlations, and feature patterns with Seaborn."),
        ("3", "⚙️ Preprocessing",  "Full cleaning pipeline: encoding, scaling, imputation, outliers, SMOTE."),
        ("4", "🤖 Model Selection", "Choose and train Classification, Regression, or Clustering algorithms."),
        ("5", "📈 Evaluation",      "Review confusion matrix, ROC curve, feature importances, and metrics."),
    ]
    for num, name, desc in steps:
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;gap:16px;padding:12px 0;border-bottom:1px solid #30363d;'>
            <div style='background:#58a6ff;color:#000;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;'>{num}</div>
            <div>
                <div style='font-weight:600;font-size:15px;'>{name}</div>
                <div style='color:#8b949e;font-size:13px;margin-top:2px;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Get Started — Upload Data"):
        st.session_state.page = "📁 1 · File Upload"
        st.rerun()
