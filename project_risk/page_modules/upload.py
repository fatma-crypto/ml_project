import streamlit as st
import pandas as pd
import io


def show():
    st.markdown("# 📁 File Upload")
    st.markdown("#### Upload your project risk dataset — CSV or Excel supported")
    st.divider()

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Drop your file here or click to browse",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: CSV (.csv), Excel (.xlsx, .xls)"
        )

        if uploaded is not None:
            try:
                # Load based on extension
                ext = uploaded.name.split(".")[-1].lower()
                if ext == "csv":
                    df = pd.read_csv(uploaded)
                    fmt = "CSV"
                else:
                    df = pd.read_excel(uploaded)
                    fmt = "Excel"

                # Store in session
                st.session_state.uploaded_df = df
                st.session_state.processed_df = None  # reset processed on new upload

                # Success banner
                st.success(f"✅ File uploaded successfully! **{uploaded.name}** ({fmt}) — {len(df):,} rows × {len(df.columns)} columns")

                # Preview
                st.markdown("### Preview (first 5 rows)")
                st.dataframe(df.head(), use_container_width=True)

                # Column summary
                st.markdown("### Column Summary")
                dtypes = df.dtypes.reset_index()
                dtypes.columns = ["Column", "Type"]
                nulls = df.isnull().sum().reset_index()
                nulls.columns = ["Column", "Nulls"]
                summary = dtypes.merge(nulls, on="Column")
                summary["Unique"] = [df[c].nunique() for c in summary["Column"]]
                st.dataframe(summary, use_container_width=True)

                # Quick stats
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", f"{len(df):,}")
                c2.metric("Columns", len(df.columns))
                num_cols = df.select_dtypes(include="number").shape[1]
                c3.metric("Numeric", num_cols)
                c4.metric("Categorical", len(df.columns) - num_cols)

                st.markdown("---")
                if st.button("➡️ Continue to Visualization"):
                    st.session_state.page = "📊 2 · Visualization"
                    st.rerun()

            except Exception as e:
                st.error(f"❌ Could not read file: {e}")

        else:
            # Show existing data if already uploaded
            if st.session_state.uploaded_df is not None:
                df = st.session_state.uploaded_df
                st.info(f"ℹ️ Using previously uploaded data — {len(df):,} rows × {len(df.columns)} columns")
                st.dataframe(df.head(), use_container_width=True)
                if st.button("➡️ Continue to Visualization"):
                    st.session_state.page = "📊 2 · Visualization"
                    st.rerun()
            else:
                st.markdown("""
                <div style='text-align:center;padding:60px 20px;color:#8b949e;'>
                    <div style='font-size:48px;margin-bottom:16px;'>📂</div>
                    <div style='font-size:16px;'>Upload a CSV or Excel file to begin</div>
                    <div style='font-size:13px;margin-top:8px;'>Max recommended size: 200 MB</div>
                </div>
                """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class='risk-card'>
            <h3 style='color:#58a6ff;'>Supported Formats</h3>
            <div style='margin-top:12px;'>
                <div style='display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid #30363d;'>
                    <span style='font-size:18px;'>📄</span>
                    <div>
                        <div style='font-size:14px;font-weight:600;'>.csv</div>
                        <div style='font-size:12px;color:#8b949e;'>Comma-separated values</div>
                    </div>
                </div>
                <div style='display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid #30363d;'>
                    <span style='font-size:18px;'>📊</span>
                    <div>
                        <div style='font-size:14px;font-weight:600;'>.xlsx</div>
                        <div style='font-size:12px;color:#8b949e;'>Excel 2007+ format</div>
                    </div>
                </div>
                <div style='display:flex;align-items:center;gap:8px;padding:8px 0;'>
                    <span style='font-size:18px;'>📈</span>
                    <div>
                        <div style='font-size:14px;font-weight:600;'>.xls</div>
                        <div style='font-size:12px;color:#8b949e;'>Legacy Excel format</div>
                    </div>
                </div>
            </div>
        </div>
        <div class='risk-card' style='margin-top:12px;'>
            <h3 style='color:#58a6ff;'>Tips</h3>
            <ul style='color:#8b949e;font-size:13px;margin-top:8px;padding-left:16px;line-height:1.8;'>
                <li>First row should be headers</li>
                <li>Include a target/label column</li>
                <li>UTF-8 encoding recommended</li>
                <li>Remove extra whitespace in headers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
