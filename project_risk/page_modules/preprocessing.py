import streamlit as st
import pandas as pd
import numpy as np


def show():
    st.markdown("# ⚙️ Preprocessing")
    st.markdown("#### Clean, encode, scale, and transform your dataset")
    st.divider()

    df_raw = st.session_state.get("uploaded_df")
    if df_raw is None:
        st.warning("⚠️ No data loaded. Please upload a file first.")
        if st.button("Go to Upload"):
            st.session_state.page = "📁 1 · File Upload"
            st.rerun()
        return

    # Work on a copy so we can reset
    if "proc_working_df" not in st.session_state:
        st.session_state.proc_working_df = df_raw.copy()

    df = st.session_state.proc_working_df
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        tabs = st.tabs([
            "🔤 Encoding", "📏 Normalization", "❓ Missing Values",
            "📐 Outliers", "🔄 Transformation", "🎯 Feature Selection", "⚖️ Imbalanced Data"
        ])

        # ── Encoding ──────────────────────────────────────────────────────
        with tabs[0]:
            st.markdown("### Encoding")
            if not cat_cols:
                st.info("No categorical columns detected.")
            else:
                enc_cols   = st.multiselect("Select columns to encode", cat_cols, default=cat_cols[:min(3, len(cat_cols))], key="enc_cols")
                enc_method = st.radio("Encoding method", ["Label Encoder", "One-Hot Encoder"], horizontal=True, key="enc_method")

                if st.button("Apply Encoding", key="btn_enc"):
                    df_work = st.session_state.proc_working_df.copy()
                    if enc_method == "Label Encoder":
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        for c in enc_cols:
                            if c in df_work.columns:
                                df_work[c] = le.fit_transform(df_work[c].astype(str))
                        st.success(f"✅ Label-encoded {len(enc_cols)} column(s).")
                    else:
                        df_work = pd.get_dummies(df_work, columns=[c for c in enc_cols if c in df_work.columns], drop_first=False)
                        st.success(f"✅ One-hot encoded → {df_work.shape[1]} columns now.")
                    st.session_state.proc_working_df = df_work
                    st.rerun()

        # ── Normalization ─────────────────────────────────────────────────
        with tabs[1]:
            st.markdown("### Normalization / Scaling")
            cur_num = st.session_state.proc_working_df.select_dtypes(include="number").columns.tolist()
            if not cur_num:
                st.info("No numeric columns found.")
            else:
                scale_cols   = st.multiselect("Select columns to scale", cur_num, default=cur_num[:min(5, len(cur_num))], key="scale_cols")
                scale_method = st.radio("Scaler", ["Standard Scaler (Z-score)", "MinMax Scaler (0–1)"], horizontal=True, key="scale_method")

                if st.button("Apply Scaling", key="btn_scale"):
                    df_work = st.session_state.proc_working_df.copy()
                    valid = [c for c in scale_cols if c in df_work.columns]
                    if scale_method.startswith("Standard"):
                        from sklearn.preprocessing import StandardScaler
                        df_work[valid] = StandardScaler().fit_transform(df_work[valid])
                        st.success(f"✅ Standard-scaled {len(valid)} column(s).")
                    else:
                        from sklearn.preprocessing import MinMaxScaler
                        df_work[valid] = MinMaxScaler().fit_transform(df_work[valid])
                        st.success(f"✅ MinMax-scaled {len(valid)} column(s).")
                    st.session_state.proc_working_df = df_work
                    st.rerun()

        # ── Missing Values ─────────────────────────────────────────────────
        with tabs[2]:
            st.markdown("### Missing Value Imputation")
            cur_df = st.session_state.proc_working_df
            missing = cur_df.isnull().sum()
            missing = missing[missing > 0]

            if missing.empty:
                st.success("✅ No missing values detected in current data.")
            else:
                st.markdown(f"**{len(missing)} column(s) have missing values:**")
                st.dataframe(missing.rename("Missing Count").reset_index(), use_container_width=True)

                impute_cols   = st.multiselect("Columns to impute", missing.index.tolist(), default=missing.index.tolist()[:3], key="imp_cols")
                impute_method = st.selectbox("Imputation method", [
                    "Simple Imputer — Mean", "Simple Imputer — Median",
                    "Simple Imputer — Most Frequent", "KNN Imputer", "Iterative Imputer"
                ], key="imp_method")

                if st.button("Apply Imputation", key="btn_imp"):
                    df_work = st.session_state.proc_working_df.copy()
                    valid = [c for c in impute_cols if c in df_work.columns]
                    try:
                        if impute_method.startswith("Simple"):
                            from sklearn.impute import SimpleImputer
                            strategy_map = {
                                "Simple Imputer — Mean": "mean",
                                "Simple Imputer — Median": "median",
                                "Simple Imputer — Most Frequent": "most_frequent"
                            }
                            imp = SimpleImputer(strategy=strategy_map[impute_method])
                            df_work[valid] = imp.fit_transform(df_work[valid])
                        elif impute_method == "KNN Imputer":
                            from sklearn.impute import KNNImputer
                            imp = KNNImputer(n_neighbors=5)
                            df_work[valid] = imp.fit_transform(df_work[valid])
                        else:
                            from sklearn.impute import IterativeImputer
                            imp = IterativeImputer(max_iter=10, random_state=42)
                            df_work[valid] = imp.fit_transform(df_work[valid])
                        st.success(f"✅ Imputed {len(valid)} column(s) using {impute_method}.")
                        st.session_state.proc_working_df = df_work
                        st.rerun()
                    except Exception as e:
                        st.error(f"Imputation failed: {e}")

        # ── Outliers ───────────────────────────────────────────────────────
        with tabs[3]:
            st.markdown("### Outlier Handling")
            cur_num2 = st.session_state.proc_working_df.select_dtypes(include="number").columns.tolist()
            if not cur_num2:
                st.info("No numeric columns found.")
            else:
                out_cols   = st.multiselect("Columns to process", cur_num2, default=cur_num2[:min(3, len(cur_num2))], key="out_cols")
                out_method = st.selectbox("Method", ["IQR", "Z-score", "Winsorization", "Clipping"], key="out_method")

                if out_method == "Clipping":
                    low_pct = st.slider("Lower clip percentile", 0.0, 10.0, 1.0, key="clip_lo")
                    hi_pct  = st.slider("Upper clip percentile", 90.0, 100.0, 99.0, key="clip_hi")

                if st.button("Apply Outlier Handling", key="btn_out"):
                    df_work = st.session_state.proc_working_df.copy()
                    valid   = [c for c in out_cols if c in df_work.columns]
                    n_before = len(df_work)
                    try:
                        if out_method == "IQR":
                            mask = pd.Series([True] * len(df_work))
                            for c in valid:
                                Q1, Q3 = df_work[c].quantile(0.25), df_work[c].quantile(0.75)
                                IQR = Q3 - Q1
                                mask &= (df_work[c] >= Q1 - 1.5*IQR) & (df_work[c] <= Q3 + 1.5*IQR)
                            df_work = df_work[mask]
                            st.success(f"✅ IQR: removed {n_before - len(df_work)} outlier rows.")

                        elif out_method == "Z-score":
                            from scipy import stats
                            z_scores = np.abs(stats.zscore(df_work[valid]))
                            mask = (z_scores < 3).all(axis=1)
                            df_work = df_work[mask]
                            st.success(f"✅ Z-score: removed {n_before - len(df_work)} rows.")

                        elif out_method == "Winsorization":
                            from scipy.stats.mstats import winsorize
                            for c in valid:
                                df_work[c] = winsorize(df_work[c], limits=[0.05, 0.05])
                            st.success(f"✅ Winsorized {len(valid)} column(s) at 5th/95th percentile.")

                        else:  # Clipping
                            for c in valid:
                                lo = np.percentile(df_work[c], low_pct)
                                hi = np.percentile(df_work[c], hi_pct)
                                df_work[c] = df_work[c].clip(lo, hi)
                            st.success(f"✅ Clipped {len(valid)} column(s).")

                        st.session_state.proc_working_df = df_work
                        st.rerun()
                    except Exception as e:
                        st.error(f"Outlier handling failed: {e}")

        # ── Feature Transformation ─────────────────────────────────────────
        with tabs[4]:
            st.markdown("### Feature Transformation")
            cur_num3 = st.session_state.proc_working_df.select_dtypes(include="number").columns.tolist()
            if not cur_num3:
                st.info("No numeric columns found.")
            else:
                t_cols   = st.multiselect("Columns to transform", cur_num3, default=cur_num3[:min(2, len(cur_num3))], key="t_cols")
                t_method = st.selectbox("Transformation", ["Log (log1p)", "Box-Cox", "Power (Yeo-Johnson)", "Polynomial Features"], key="t_method")

                if t_method == "Polynomial Features":
                    poly_deg = st.slider("Polynomial degree", 2, 4, 2, key="poly_deg")

                if st.button("Apply Transformation", key="btn_trans"):
                    df_work = st.session_state.proc_working_df.copy()
                    valid   = [c for c in t_cols if c in df_work.columns]
                    try:
                        if t_method == "Log (log1p)":
                            for c in valid:
                                df_work[f"{c}_log"] = np.log1p(df_work[c].clip(lower=0))
                            st.success(f"✅ Log-transformed {len(valid)} column(s) (new columns added).")

                        elif t_method == "Box-Cox":
                            from sklearn.preprocessing import PowerTransformer
                            pt = PowerTransformer(method="box-cox")
                            vals = df_work[valid].values
                            # Box-Cox requires positive values
                            vals = vals - vals.min(axis=0) + 1
                            df_work[valid] = pt.fit_transform(vals)
                            st.success(f"✅ Box-Cox transformed {len(valid)} column(s).")

                        elif t_method == "Power (Yeo-Johnson)":
                            from sklearn.preprocessing import PowerTransformer
                            pt = PowerTransformer(method="yeo-johnson")
                            df_work[valid] = pt.fit_transform(df_work[valid])
                            st.success(f"✅ Yeo-Johnson transformed {len(valid)} column(s).")

                        else:  # Polynomial
                            from sklearn.preprocessing import PolynomialFeatures
                            pf = PolynomialFeatures(degree=poly_deg, include_bias=False)
                            poly_arr  = pf.fit_transform(df_work[valid])
                            poly_names = pf.get_feature_names_out(valid)
                            poly_df   = pd.DataFrame(poly_arr, columns=poly_names, index=df_work.index)
                            df_work   = pd.concat([df_work.drop(columns=valid), poly_df], axis=1)
                            st.success(f"✅ Polynomial degree-{poly_deg} features: {poly_arr.shape[1]} new columns.")

                        st.session_state.proc_working_df = df_work
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transformation failed: {e}")

        # ── Feature Selection ──────────────────────────────────────────────
        with tabs[5]:
            st.markdown("### Feature Selection & Dimensionality Reduction")
            cur_num4 = st.session_state.proc_working_df.select_dtypes(include="number").columns.tolist()

            fs_method = st.radio("Method", ["RFE (Recursive Feature Elimination)", "PCA"], horizontal=True, key="fs_method")

            target_col = st.selectbox("Target column", st.session_state.proc_working_df.columns.tolist(), key="fs_target")
            feature_candidates = [c for c in cur_num4 if c != target_col]

            if fs_method == "RFE (Recursive Feature Elimination)":
                n_features = st.slider("Number of features to select", 2, max(2, len(feature_candidates)), min(10, len(feature_candidates)), key="rfe_n")
                if st.button("Apply RFE", key="btn_rfe"):
                    try:
                        from sklearn.feature_selection import RFE
                        from sklearn.ensemble import RandomForestClassifier
                        df_work = st.session_state.proc_working_df.copy()
                        X = df_work[feature_candidates].fillna(0)
                        y = df_work[target_col].fillna(0)
                        rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=n_features)
                        rfe.fit(X, y)
                        selected = [f for f, s in zip(feature_candidates, rfe.support_) if s]
                        df_work = df_work[selected + [target_col]]
                        st.success(f"✅ RFE selected {len(selected)} features: {selected}")
                        st.session_state.proc_working_df = df_work
                        st.session_state.target_col = target_col
                        st.rerun()
                    except Exception as e:
                        st.error(f"RFE failed: {e}")
            else:
                n_components = st.slider("PCA components", 2, max(2, len(feature_candidates)), min(5, len(feature_candidates)), key="pca_n")
                if st.button("Apply PCA", key="btn_pca"):
                    try:
                        from sklearn.decomposition import PCA
                        df_work = st.session_state.proc_working_df.copy()
                        X = df_work[feature_candidates].fillna(0)
                        pca = PCA(n_components=n_components, random_state=42)
                        pca_arr = pca.fit_transform(X)
                        pca_df  = pd.DataFrame(pca_arr, columns=[f"PC{i+1}" for i in range(n_components)], index=df_work.index)
                        var_exp = pca.explained_variance_ratio_.sum()
                        df_work = pd.concat([pca_df, df_work[[target_col]].reset_index(drop=True)], axis=1)
                        st.success(f"✅ PCA → {n_components} components explaining {var_exp:.1%} variance.")
                        st.session_state.proc_working_df = df_work
                        st.session_state.target_col = target_col
                        st.rerun()
                    except Exception as e:
                        st.error(f"PCA failed: {e}")

        # ── Imbalanced Data ────────────────────────────────────────────────
        with tabs[6]:
            st.markdown("### Handling Imbalanced Data")
            target_col_imb = st.selectbox("Target column", st.session_state.proc_working_df.columns.tolist(), key="imb_target")
            imb_method     = st.radio("Method", ["SMOTE (Oversampling)", "Random Undersampling"], horizontal=True, key="imb_method")

            cur_num5 = st.session_state.proc_working_df.select_dtypes(include="number").columns.tolist()
            feature_cols_imb = [c for c in cur_num5 if c != target_col_imb]

            if st.button("Apply Balancing", key="btn_imb"):
                try:
                    df_work = st.session_state.proc_working_df.copy()
                    X = df_work[feature_cols_imb].fillna(0)
                    y = df_work[target_col_imb].fillna(0)
                    y_int = y.astype(int)

                    if imb_method == "SMOTE (Oversampling)":
                        from imblearn.over_sampling import SMOTE
                        sm = SMOTE(random_state=42)
                        X_res, y_res = sm.fit_resample(X, y_int)
                    else:
                        from imblearn.under_sampling import RandomUnderSampler
                        rus = RandomUnderSampler(random_state=42)
                        X_res, y_res = rus.fit_resample(X, y_int)

                    df_res = pd.DataFrame(X_res, columns=feature_cols_imb)
                    df_res[target_col_imb] = y_res
                    orig_dist = y_int.value_counts().to_dict()
                    new_dist  = pd.Series(y_res).value_counts().to_dict()
                    st.success(f"✅ {imb_method} applied. Rows: {len(df_work)} → {len(df_res)}")
                    st.write(f"Before: {orig_dist} | After: {new_dist}")
                    st.session_state.proc_working_df = df_res
                    st.rerun()
                except ImportError:
                    st.error("Install `imbalanced-learn`: `pip install imbalanced-learn`")
                except Exception as e:
                    st.error(f"Balancing failed: {e}")

    # ── Right Panel: state summary ─────────────────────────────────────────
    with col_right:
        cur_df = st.session_state.proc_working_df
        st.markdown("""<div class='risk-card'>""", unsafe_allow_html=True)
        st.markdown("### 📋 Current State")
        st.metric("Rows", f"{len(cur_df):,}")
        st.metric("Columns", len(cur_df.columns))
        n_num = cur_df.select_dtypes(include="number").shape[1]
        st.metric("Numeric", n_num)
        st.metric("Missing Values", int(cur_df.isnull().sum().sum()))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reset to Original", key="btn_reset"):
            st.session_state.proc_working_df = st.session_state.uploaded_df.copy()
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Save & Continue to Model", key="btn_save_proc"):
            st.session_state.processed_df = st.session_state.proc_working_df.copy()
            st.success("Preprocessing saved!")
            st.session_state.page = "🤖 4 · Model"
            st.rerun()

        st.markdown("### Column Preview")
        st.write(list(cur_df.columns[:10]))
        if len(cur_df.columns) > 10:
            st.caption(f"…and {len(cur_df.columns)-10} more")
