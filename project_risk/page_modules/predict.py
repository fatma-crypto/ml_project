import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


@st.cache_resource
def load_pretrained():
    model_path = "best_model.joblib"
    data_path  = "project_risk_raw_dataset.csv"
    if os.path.exists(model_path) and os.path.exists(data_path):
        model = joblib.load(model_path)
        df    = pd.read_csv(data_path)
        return model, df
    return None, None


def generate_explanation(input_df, prediction, full_df):
    if prediction == 0:
        return "✅ **Low Risk** — The parameters align with historically successful projects. Continue monitoring timeline and budget."

    explanation = ["**High Risk Factors Detected:**\n"]
    high_df = full_df[full_df["Risk_Level"] == "High"] if "Risk_Level" in full_df.columns else full_df
    low_df  = full_df[full_df["Risk_Level"] == "Low"]  if "Risk_Level" in full_df.columns else full_df

    features_to_check = [
        "Schedule_Pressure", "Budget_Utilization_Rate", "Team_Turnover_Rate",
        "Complexity_Score", "Technical_Debt_Level", "Change_Request_Frequency"
    ]
    factors = []
    for feat in features_to_check:
        if feat not in input_df.columns:
            continue
        val = input_df[feat].iloc[0]
        try:
            avg_high = high_df[feat].median()
            avg_low  = low_df[feat].median()
            if abs(val - avg_high) < abs(val - avg_low):
                factors.append(f"- **{feat}**: `{val:.2f}` (typical low-risk level: `{avg_low:.2f}`)")
        except Exception:
            pass

    if factors:
        explanation.extend(factors)
        explanation.append("\n\n💡 **Suggestion**: Allocate more resources, extend the timeline, or reduce scope to mitigate these factors.")
    else:
        explanation.append("Risk pattern is complex — likely driven by categorical factors like Project Type or Methodology.")
    return "\n".join(explanation)


def show():
    st.markdown("# 🔮 Risk Prediction")
    st.markdown("#### Predict project risk level using the trained model")
    st.divider()

    # Priority: session model → pre-trained file
    model = st.session_state.get("trained_model")
    df    = None

    if model is not None:
        st.success("✅ Using model trained in this session.")
        df = st.session_state.get("processed_df") or st.session_state.get("uploaded_df")
    else:
        model, df = load_pretrained()
        if model is not None:
            st.info("ℹ️ Using pre-trained model (`best_model.joblib`).")
        else:
            st.error("❌ No model found. Please train a model on the **🤖 Model** page, or place `best_model.joblib` in the app directory.")
            return

    if df is None:
        st.error("No dataset available to read input ranges.")
        return

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Remove target/ID columns from inputs
    exclude = {"Risk_Level", "Risk_Binary", "Project_ID"}
    input_num = [c for c in num_cols if c not in exclude]
    input_cat = [c for c in cat_cols if c not in exclude]

    st.markdown("### 📋 Enter Project Details")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Categorical Fields")
        cat_inputs = {}
        for col in input_cat:
            options = df[col].dropna().unique().tolist()
            cat_inputs[col] = st.selectbox(col.replace("_", " "), options, key=f"pred_cat_{col}")

    with col_b:
        st.markdown("#### Numeric Fields")
        num_inputs = {}
        for col in input_num:
            median_val = float(df[col].median())
            min_val    = float(df[col].min())
            max_val    = float(df[col].max())
            num_inputs[col] = st.number_input(
                col.replace("_", " "), value=median_val,
                min_value=min_val, max_value=max_val, key=f"pred_num_{col}"
            )

    st.divider()

    if st.button("🔍 Analyze Risk", type="primary", key="btn_predict"):
        input_data = {**cat_inputs, **num_inputs}
        input_df   = pd.DataFrame([input_data])

        # Add derived features if using pre-trained model (matches original train pipeline)
        if "Project_Budget_USD" in input_df and "Estimated_Timeline_Months" in input_df:
            input_df["Budget_per_Month"]  = input_df["Project_Budget_USD"] / input_df["Estimated_Timeline_Months"].replace(0, 1)
        if "Complexity_Score" in input_df and "Team_Size" in input_df:
            input_df["Workload_Index"]    = input_df["Complexity_Score"] / input_df["Team_Size"].replace(0, 1)
        for col in ["Tech_Environment_Stability_Missing", "Change_Control_Maturity_Missing", "Risk_Management_Maturity_Missing"]:
            if col not in input_df.columns:
                input_df[col] = 0

        try:
            prediction  = model.predict(input_df)[0]
            has_proba   = hasattr(model, "predict_proba")
            probability = model.predict_proba(input_df)[0][1] if has_proba else None

            c1, c2 = st.columns([1, 2])
            with c1:
                if prediction == 1 or prediction == "High":
                    st.error("## 🔴 HIGH RISK")
                else:
                    st.success("## 🟢 LOW RISK")
                if probability is not None:
                    st.metric("Risk Probability", f"{probability:.1%}")

            with c2:
                st.markdown("### 🤖 AI Analysis")
                explanation = generate_explanation(input_df, int(prediction == 1 or prediction == "High"), df)
                st.markdown(explanation)

            # Feature importance chart if available
            if os.path.exists("feature_importance.png"):
                st.divider()
                st.markdown("### 🔍 Model Feature Importances")
                st.image("feature_importance.png", use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.caption("Tip: If using a pre-trained model, make sure the input features match the training schema.")

    # ── Chat Section ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💬 Risk Chatbot")
    user_input = st.text_input("Ask about risk factors:", placeholder="Why might this project be high risk?", key="chat_input")
    if user_input:
        keywords = ["why", "explain", "reason", "what", "factor", "cause", "high", "low"]
        if any(k in user_input.lower() for k in keywords):
            st.chat_message("assistant").write(
                "Risk is primarily driven by factors like **schedule pressure**, **budget utilization**, "
                "**team turnover**, and **technical complexity**. Check the Feature Importance chart for your "
                "model's specific top predictors. Run the analysis above for project-specific insights."
            )
        else:
            st.chat_message("assistant").write(
                "I focus on project risk assessment. Try asking: *'What makes this project high risk?'* "
                "or *'What are the key risk factors?'*"
            )
