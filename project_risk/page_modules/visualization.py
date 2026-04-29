import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "#1c2128",
        "axes.facecolor":   "#1c2128",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "text.color":       "#e6edf3",
        "grid.color":       "#30363d",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
        "axes.titlecolor":  "#e6edf3",
    })
    sns.set_palette("tab10")


def show():
    st.markdown("# 📊 Data Visualization")
    st.markdown("#### Explore distributions, relationships, and feature patterns")
    st.divider()

    df = st.session_state.get("uploaded_df")
    if df is None:
        st.warning("⚠️ No data loaded. Please upload a file first.")
        if st.button("Go to Upload"):
            st.session_state.page = "📁 1 · File Upload"
            st.rerun()
        return

    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include="object").columns.tolist()
    all_cols  = df.columns.tolist()

    _apply_style()

    tab_line, tab_scatter, tab_box, tab_heatmap, tab_count = st.tabs([
        "📈 Line Plot", "🔵 Scatter Plot", "📦 Box Plot", "🟥 Heatmap", "📊 Count Plot"
    ])

    # ── Line Plot ──────────────────────────────────────────────────────────
    with tab_line:
        st.markdown("### Line Plot")
        st.caption("Best for numeric columns ordered by index or a sequence column.")

        if not num_cols:
            st.info("No numeric columns found for line plot.")
        else:
            c1, c2, c3 = st.columns(3)
            y_col   = c1.selectbox("Y-axis (numeric)", num_cols, key="line_y")
            x_col   = c2.selectbox("X-axis (optional index/numeric)", ["Index"] + num_cols, key="line_x")
            hue_col = c3.selectbox("Color by (optional)", ["None"] + cat_cols, key="line_hue")

            sample_n = st.slider("Sample size (for speed)", 100, min(5000, len(df)), min(1000, len(df)), key="line_sample")
            plot_df = df.sample(min(sample_n, len(df)), random_state=42).reset_index(drop=True)

            if st.button("Generate Line Plot", key="btn_line"):
                fig, ax = plt.subplots(figsize=(10, 4))
                x_data = plot_df.index if x_col == "Index" else plot_df[x_col]
                if hue_col != "None":
                    for grp, gdf in plot_df.groupby(hue_col):
                        xg = gdf.index if x_col == "Index" else gdf[x_col]
                        ax.plot(xg, gdf[y_col], label=str(grp), linewidth=1.5)
                    ax.legend(title=hue_col, fontsize=9)
                else:
                    ax.plot(x_data, plot_df[y_col], color="#58a6ff", linewidth=1.5)
                ax.set_xlabel(x_col, fontsize=11)
                ax.set_ylabel(y_col, fontsize=11)
                ax.set_title(f"{y_col} over {x_col}", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Scatter Plot ───────────────────────────────────────────────────────
    with tab_scatter:
        st.markdown("### Scatter Plot")
        st.caption("Reveals relationships and correlations between two numeric features.")

        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for a scatter plot.")
        else:
            c1, c2, c3 = st.columns(3)
            x_col   = c1.selectbox("X-axis", num_cols, key="sc_x")
            y_col   = c2.selectbox("Y-axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
            hue_col = c3.selectbox("Color by (optional)", ["None"] + cat_cols, key="sc_hue")

            sample_n = st.slider("Sample size", 100, min(3000, len(df)), min(1000, len(df)), key="sc_sample")
            alpha    = st.slider("Point opacity", 0.1, 1.0, 0.6, key="sc_alpha")

            if st.button("Generate Scatter Plot", key="btn_scatter"):
                plot_df = df.sample(min(sample_n, len(df)), random_state=42)
                fig, ax = plt.subplots(figsize=(9, 5))
                hue_data = plot_df[hue_col] if hue_col != "None" else None
                sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_data, alpha=alpha, ax=ax, s=40)
                ax.set_title(f"{x_col} vs {y_col}", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3)
                if hue_col != "None":
                    ax.legend(title=hue_col, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Box Plot ───────────────────────────────────────────────────────────
    with tab_box:
        st.markdown("### Box Plot")
        st.caption("Compare distributions — works for both numeric and grouped categorical data.")

        c1, c2 = st.columns(2)
        y_col = c1.selectbox("Numeric column (Y)", num_cols if num_cols else all_cols, key="box_y")
        x_col = c2.selectbox("Group by (X, optional)", ["None"] + cat_cols, key="box_x")

        if st.button("Generate Box Plot", key="btn_box"):
            fig, ax = plt.subplots(figsize=(10, 5))
            if x_col == "None":
                sns.boxplot(y=df[y_col], ax=ax, color="#58a6ff", width=0.4)
                ax.set_title(f"Distribution of {y_col}", fontsize=13, fontweight="bold")
            else:
                # Limit categories for readability
                top_cats = df[x_col].value_counts().head(15).index
                plot_df = df[df[x_col].isin(top_cats)]
                sns.boxplot(data=plot_df, x=x_col, y=y_col, ax=ax, palette="tab10")
                ax.set_title(f"{y_col} by {x_col}", fontsize=13, fontweight="bold")
                plt.xticks(rotation=30, ha="right", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Heatmap ────────────────────────────────────────────────────────────
    with tab_heatmap:
        st.markdown("### Correlation Heatmap")
        st.caption("Shows pairwise correlation between numeric features.")

        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            selected_nums = st.multiselect(
                "Select numeric columns (leave empty for all)",
                num_cols, default=num_cols[:min(12, len(num_cols))],
                key="heat_cols"
            )
            annot = st.checkbox("Show values", value=True, key="heat_annot")

            if st.button("Generate Heatmap", key="btn_heat"):
                cols_to_use = selected_nums if selected_nums else num_cols
                corr = df[cols_to_use].corr()
                fig, ax = plt.subplots(figsize=(max(8, len(cols_to_use)), max(6, len(cols_to_use) * 0.7)))
                sns.heatmap(
                    corr, annot=annot, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, linecolor="#30363d",
                    annot_kws={"size": 8}, ax=ax
                )
                ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Count Plot (categorical) ───────────────────────────────────────────
    with tab_count:
        st.markdown("### Count Plot")
        st.caption("Shows value frequencies — ideal for categorical columns.")

        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            c1, c2 = st.columns(2)
            col    = c1.selectbox("Categorical column", cat_cols, key="cnt_col")
            hue    = c2.selectbox("Color by (optional)", ["None"] + cat_cols, key="cnt_hue")

            top_n = st.slider("Show top N categories", 3, 30, 10, key="cnt_topn")

            if st.button("Generate Count Plot", key="btn_count"):
                top_vals = df[col].value_counts().head(top_n).index
                plot_df  = df[df[col].isin(top_vals)]
                fig, ax  = plt.subplots(figsize=(10, 4))
                hue_data = plot_df[hue] if hue != "None" else None
                sns.countplot(data=plot_df, x=col, hue=hue_data, ax=ax, palette="tab10",
                              order=top_vals)
                ax.set_title(f"Count of {col}", fontsize=13, fontweight="bold")
                plt.xticks(rotation=30, ha="right", fontsize=9)
                ax.grid(True, alpha=0.3, axis="y")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
