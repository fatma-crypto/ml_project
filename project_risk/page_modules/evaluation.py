import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


def _style():
    plt.rcParams.update({
        "figure.facecolor": "#1c2128", "axes.facecolor": "#1c2128",
        "axes.edgecolor": "#30363d", "axes.labelcolor": "#e6edf3",
        "xtick.color": "#8b949e", "ytick.color": "#8b949e",
        "text.color": "#e6edf3", "grid.color": "#30363d",
    })


def show():
    st.markdown("# 📈 Model Evaluation")
    st.markdown("#### Performance metrics, confusion matrix, and feature importances")
    st.divider()

    _style()

    # ── Check for pre-trained assets OR session model ─────────────────────
    has_pretrained = os.path.exists("confusion_matrix.png") and os.path.exists("roc_curve.png")
    has_session    = st.session_state.get("trained_model") is not None and st.session_state.get("y_test") is not None

    if not has_pretrained and not has_session:
        st.info("ℹ️ No evaluation data found. Train a model first on the **🤖 Model** page.")
        if os.path.exists("training_results.csv"):
            st.markdown("### Pre-existing Training Results")
            results_df = pd.read_csv("training_results.csv")
            st.dataframe(results_df, use_container_width=True)
        return

    # Tabs
    tab_metrics, tab_cm, tab_roc, tab_fi, tab_comparison = st.tabs([
        "📊 Metrics", "🟦 Confusion Matrix", "📈 ROC Curve",
        "🔍 Feature Importance", "🏆 Model Comparison"
    ])

    # ── Metrics ────────────────────────────────────────────────────────────
    with tab_metrics:
        if has_session:
            from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                         recall_score, classification_report,
                                         mean_squared_error, r2_score)
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            task   = st.session_state.get("model_task", "Classification")

            if task == "Classification":
                acc  = accuracy_score(y_test, y_pred)
                f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{acc:.3f}")
                c2.metric("F1 Score",  f"{f1:.3f}")
                c3.metric("Precision", f"{prec:.3f}")
                c4.metric("Recall",    f"{rec:.3f}")

                st.markdown("### Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).T
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            else:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2   = r2_score(y_test, y_pred)
                mae  = np.mean(np.abs(y_test - y_pred))
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{rmse:.4f}")
                c2.metric("R²",   f"{r2:.4f}")
                c3.metric("MAE",  f"{mae:.4f}")
        else:
            st.info("Session metrics not available. View pre-trained results in other tabs.")

    # ── Confusion Matrix ───────────────────────────────────────────────────
    with tab_cm:
        if has_session and st.session_state.get("model_task") == "Classification":
            from sklearn.metrics import confusion_matrix
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        linewidths=0.5, linecolor="#30363d", ax=ax)
            ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        elif has_pretrained:
            st.markdown("### Pre-trained Model — Confusion Matrix")
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("Train a classification model to view the confusion matrix.")

    # ── ROC Curve ─────────────────────────────────────────────────────────
    with tab_roc:
        plotted = False
        if has_session and st.session_state.get("model_task") == "Classification":
            model  = st.session_state.trained_model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            if hasattr(model, "predict_proba"):
                from sklearn.metrics import roc_auc_score, roc_curve
                try:
                    y_prob = model.predict_proba(X_test)
                    classes = np.unique(y_test)
                    if len(classes) == 2:
                        y_prob_pos = y_prob[:, 1]
                        auc = roc_auc_score(y_test, y_prob_pos)
                        fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.plot(fpr, tpr, color="#58a6ff", linewidth=2, label=f"AUC = {auc:.3f}")
                        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.set_title("ROC Curve", fontsize=13, fontweight="bold")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        plotted = True
                except Exception:
                    pass

        if not plotted and has_pretrained:
            st.markdown("### Pre-trained Model — ROC Curve")
            st.image("roc_curve.png", use_container_width=True)
        elif not plotted:
            st.info("ROC curve requires a binary classification model with `predict_proba`.")

    # ── Feature Importance ─────────────────────────────────────────────────
    with tab_fi:
        plotted_fi = False
        if has_session:
            model    = st.session_state.trained_model
            features = st.session_state.get("feature_cols", [])
            imp = None
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            elif hasattr(model, "coef_"):
                coef = model.coef_
                imp = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

            if imp is not None and len(imp) == len(features):
                fi_df = pd.DataFrame({"Feature": features, "Importance": imp})
                fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(9, max(4, len(fi_df) * 0.35)))
                sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax, palette="viridis")
                ax.set_title("Top Feature Importances", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                plotted_fi = True

        if not plotted_fi and has_pretrained and os.path.exists("feature_importance.png"):
            st.markdown("### Pre-trained Model — Feature Importances")
            st.image("feature_importance.png", use_container_width=True)
        elif not plotted_fi:
            st.info("Feature importances are available for tree-based and linear models.")

    # ── Model Comparison ───────────────────────────────────────────────────
    with tab_comparison:
        if os.path.exists("training_results.csv"):
            st.markdown("### Model Comparison — Training Run")
            results_df = pd.read_csv("training_results.csv")
            # Display numeric columns only
            numeric_res = results_df.select_dtypes(include="number")
            st.dataframe(results_df[["Model"] + numeric_res.columns.tolist()
                                    if "Model" in results_df.columns else results_df],
                         use_container_width=True)

            # Bar chart of accuracy
            if "Accuracy" in results_df.columns and "Model" in results_df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#58a6ff" if i == results_df["Accuracy"].idxmax() else "#30363d"
                          for i in results_df.index]
                ax.bar(results_df["Model"], results_df["Accuracy"], color=colors)
                ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
                ax.set_ylabel("Accuracy")
                plt.xticks(rotation=20, ha="right")
                ax.grid(True, alpha=0.3, axis="y")
                for i, (_, row) in enumerate(results_df.iterrows()):
                    ax.text(i, row["Accuracy"] + 0.002, f"{row['Accuracy']:.3f}",
                            ha="center", va="bottom", fontsize=10, color="#e6edf3")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Train multiple models to see a comparison here. Pre-saved `training_results.csv` not found.")
