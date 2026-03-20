import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io


def _style_chart(fig, ax):
    """Apply theme-aware styling to matplotlib charts."""
    theme = st.session_state.get("theme", "dark")
    if theme == "dark":
        bg, fg, grid, acc = "#161d2e", "#f1f5f9", "#1e293b", "#00e5a0"
    else:
        bg, fg, grid, acc = "#ffffff", "#0f172a", "#e2e8f0", "#059669"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(grid)
    ax.tick_params(colors=fg, labelsize=9)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(acc)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")


def show_data_cleaner():
    raw_df = st.session_state.df

    if st.session_state.get("cleaned_df") is None:
        st.session_state.cleaned_df = raw_df.copy()

    working_df = st.session_state.cleaned_df.copy()

    st.header("Data Cleaner")
    st.markdown(
        "Handle missing values, remove duplicates, encode categoricals, "
        "scale features, and remove outliers. All changes apply to a clean "
        "copy used across all other modules.",
        unsafe_allow_html=False
    )
    st.markdown("---")

    # Summary metrics + Reset button on same row
    col_m1, col_m2, col_m3, col_rst = st.columns([2, 2, 2, 1.5])
    with col_m1: st.metric("Rows",    f"{working_df.shape[0]:,}")
    with col_m2: st.metric("Columns", working_df.shape[1])
    with col_m3:
        mp = round((working_df.isnull().sum().sum() / (working_df.shape[0] * working_df.shape[1])) * 100, 1)
        st.metric("Missing", f"{mp}%")
    with col_rst:
        st.markdown("<div style='padding-top:1.6rem;'>", unsafe_allow_html=True)
        if st.button("Reset All", key="reset_cleaner"):
            st.session_state.cleaned_df = raw_df.copy()
            st.success("Reset to original dataset.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Missing Values", "Duplicates",
        "Encoding", "Scaling", "Outliers",
    ])

    # ── Overview ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("Current Dataset Overview")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Data Types**")
            dtype_df = working_df.dtypes.reset_index()
            dtype_df.columns = ["Column", "Type"]
            dtype_df["Type"]      = dtype_df["Type"].astype(str)
            dtype_df["Missing"]   = working_df.isnull().sum().values
            dtype_df["Missing %"] = (working_df.isnull().mean()*100).round(1).values
            dtype_df["Unique"]    = working_df.nunique().values
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("**Sample Rows**")
            st.dataframe(working_df.head(8), use_container_width=True)

        missing_s = working_df.isnull().sum()
        missing_s = missing_s[missing_s > 0]
        if not missing_s.empty:
            st.markdown("**Missing Values per Column**")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(missing_s.index, missing_s.values, color="#3b82f6", alpha=0.85)
            ax.set_xlabel("Column"); ax.set_ylabel("Count"); ax.set_title("Missing Values")
            plt.xticks(rotation=45, ha="right")
            _style_chart(fig, ax); fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.success("No missing values in current dataset.")

        

    # ── Missing Values ─────────────────────────────────────────────────
    with tab2:
        st.subheader("Handle Missing Values")
        missing_cols = working_df.columns[working_df.isnull().any()].tolist()

        if not missing_cols:
            st.success("No missing values found.")
        else:
            st.info(f"Found **{len(missing_cols)}** column(s) with missing values.")

            for col in missing_cols:
                mc  = int(working_df[col].isnull().sum())
                mp  = round(working_df[col].isnull().mean()*100, 1)
                dt  = str(working_df[col].dtype)
                num = working_df[col].dtype in ["int64","float64"]
                with st.expander(f"{col}  —  {mc} missing ({mp}%)  [{dt}]"):
                    opts = ["Mean","Median","Mode","Constant (0)","Custom value","Drop rows"] if num \
                        else ["Mode","Constant (Unknown)","Custom value","Drop rows"]
                    strategy = st.selectbox("Strategy", opts, key=f"miss_{col}")
                    custom = None
                    if strategy == "Custom value":
                        custom = st.text_input("Fill value", value="0", key=f"mc_{col}")
                    if st.button(f"Apply to {col}", key=f"ma_{col}"):
                        tmp = working_df.copy()
                        try:
                            if strategy == "Mean":           tmp[col] = tmp[col].fillna(tmp[col].mean())
                            elif strategy == "Median":       tmp[col] = tmp[col].fillna(tmp[col].median())
                            elif strategy == "Mode":         tmp[col] = tmp[col].fillna(tmp[col].mode()[0])
                            elif strategy == "Constant (0)": tmp[col] = tmp[col].fillna(0)
                            elif strategy == "Constant (Unknown)": tmp[col] = tmp[col].fillna("Unknown")
                            elif strategy == "Custom value":
                                tmp[col] = tmp[col].fillna(float(custom) if num else custom)
                            elif strategy == "Drop rows":    tmp = tmp.dropna(subset=[col])
                            st.session_state.cleaned_df = tmp
                            st.success(f"Applied {strategy} to {col}.")
                            st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

            st.markdown("---")
            st.markdown("**Bulk Apply**")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Fill all numeric with Median", key="bulk_num"):
                    tmp = working_df.copy()
                    for c in tmp.select_dtypes(include=["int64","float64"]).columns:
                        tmp[c] = tmp[c].fillna(tmp[c].median())
                    st.session_state.cleaned_df = tmp
                    st.success("All numeric filled with median."); st.rerun()
            with b2:
                if st.button("Fill all categorical with Mode", key="bulk_cat"):
                    tmp = working_df.copy()
                    for c in tmp.select_dtypes(include=["object","category"]).columns:
                        if tmp[c].isnull().any(): tmp[c] = tmp[c].fillna(tmp[c].mode()[0])
                    st.session_state.cleaned_df = tmp
                    st.success("All categorical filled with mode."); st.rerun()

    # ── Duplicates ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Duplicate Row Detection")
        dup = int(working_df.duplicated().sum())
        if dup == 0:
            st.success("No duplicate rows found.")
        else:
            st.warning(f"**{dup}** duplicate rows ({round(dup/len(working_df)*100,1)}% of data).")
            st.dataframe(working_df[working_df.duplicated(keep=False)].head(10), use_container_width=True)
            keep = st.radio("Keep which copy?", ["first","last","Drop all"], horizontal=True, key="dup_keep")
            if st.button("Remove Duplicates", key="rm_dups"):
                tmp = working_df.copy()
                tmp = tmp.drop_duplicates(keep=False) if keep=="Drop all" else tmp.drop_duplicates(keep=keep)
                st.session_state.cleaned_df = tmp
                st.success(f"Removed {dup} rows. Dataset now has {len(tmp):,} rows.")
                st.rerun()

    # ── Encoding ───────────────────────────────────────────────────────
    with tab4:
        st.subheader("Categorical Encoding")
        cat_cols = working_df.select_dtypes(include=["object","category"]).columns.tolist()
        if not cat_cols:
            st.success("No categorical columns — no encoding needed.")
        else:
            st.info(f"Found **{len(cat_cols)}** categorical column(s).")
            for col in cat_cols:
                uv   = working_df[col].nunique()
                samp = working_df[col].dropna().unique()[:5].tolist()
                with st.expander(f"{col}  —  {uv} unique  |  e.g. {samp}"):
                    enc = st.selectbox("Method",
                        ["Label Encoding","One-Hot Encoding","Ordinal (custom order)","Drop column"],
                        key=f"enc_{col}")
                    ord_str = None
                    if enc == "Ordinal (custom order)":
                        all_v = sorted(working_df[col].dropna().unique().tolist())
                        ord_str = st.text_input("Order (comma-separated)",
                            value=", ".join(str(v) for v in all_v), key=f"ord_{col}")
                    if st.button(f"Encode {col}", key=f"ea_{col}"):
                        tmp = working_df.copy()
                        try:
                            if enc == "Label Encoding":
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                tmp[col] = le.fit_transform(tmp[col].astype(str))
                                st.session_state.cleaned_df = tmp
                                st.success(f"Label-encoded. Classes: {list(le.classes_)}")
                            elif enc == "One-Hot Encoding":
                                dm  = pd.get_dummies(tmp[col], prefix=col, drop_first=False)
                                tmp = pd.concat([tmp.drop(columns=[col]), dm], axis=1)
                                st.session_state.cleaned_df = tmp
                                st.success(f"One-hot encoded — {len(dm.columns)} new columns.")
                            elif enc == "Ordinal (custom order)":
                                order = [v.strip() for v in ord_str.split(",")]
                                tmp[col] = tmp[col].map({v:i for i,v in enumerate(order)})
                                st.session_state.cleaned_df = tmp
                                st.success(f"Ordinal-encoded: {order}")
                            elif enc == "Drop column":
                                tmp = tmp.drop(columns=[col])
                                st.session_state.cleaned_df = tmp
                                st.success(f"Column {col} dropped.")
                            st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

    # ── Scaling ────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Feature Scaling")
        num_cols = working_df.select_dtypes(include=["int64","float64"]).columns.tolist()
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            st.info(f"**{len(num_cols)}** numeric columns available.")
            to_scale = st.multiselect("Columns to scale", num_cols, default=num_cols, key="sc_cols")
            method   = st.radio("Method",
                ["StandardScaler (Z-score)","MinMaxScaler (0-1)","RobustScaler (IQR)"],
                horizontal=True, key="sc_method")
            if to_scale:
                prev = st.selectbox("Preview column", to_scale, key="sc_prev")
                fig, ax = plt.subplots(figsize=(6,3))
                ax.hist(working_df[prev].dropna(), bins=30, color="#8b5cf6", alpha=0.85)
                ax.set_title(f"{prev} before scaling"); ax.set_xlabel(prev); ax.set_ylabel("Frequency")
                _style_chart(fig, ax); fig.tight_layout()
                st.pyplot(fig); plt.close(fig)
            if st.button("Apply Scaling", key="apply_sc"):
                if not to_scale: st.warning("Select at least one column.")
                else:
                    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                    tmp = working_df.copy()
                    try:
                        scaler = StandardScaler() if method.startswith("Standard") else \
                                 MinMaxScaler()   if method.startswith("MinMax")   else RobustScaler()
                        tmp[to_scale] = scaler.fit_transform(tmp[to_scale])
                        st.session_state.cleaned_df = tmp
                        st.success(f"{method} applied to {len(to_scale)} columns.")
                        st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

    # ── Outliers ───────────────────────────────────────────────────────
    with tab6:
        st.subheader("Outlier Detection")
        num_cols = working_df.select_dtypes(include=["int64","float64"]).columns.tolist()
        if not num_cols:
            st.warning("No numeric columns.")
        else:
            out_col = st.selectbox("Column", num_cols, key="out_col")
            det_method = st.radio("Method", ["IQR","Z-Score (3σ)"], horizontal=True, key="out_method")
            col_data = working_df[out_col].dropna()

            if det_method == "IQR":
                Q1,Q3  = col_data.quantile(0.25), col_data.quantile(0.75)
                IQR    = Q3-Q1; lower = Q1-1.5*IQR; upper = Q3+1.5*IQR
            else:
                m,s    = col_data.mean(), col_data.std(); lower = m-3*s; upper = m+3*s

            mask  = (working_df[out_col]<lower)|(working_df[out_col]>upper)
            count = int(mask.sum())

            sc1,sc2,sc3 = st.columns(3)
            sc1.metric("Lower Bound", f"{lower:.2f}")
            sc2.metric("Upper Bound", f"{upper:.2f}")
            sc3.metric("Outliers",    count)

            fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,3.5))
            ax1.boxplot(col_data, patch_artist=True,
                boxprops=dict(facecolor="#1e293b",color="#00e5a0"),
                medianprops=dict(color="#00e5a0",linewidth=2),
                whiskerprops=dict(color="#94a3b8"), capprops=dict(color="#94a3b8"),
                flierprops=dict(marker="o",color="#ef4444",markersize=4,alpha=0.7))
            ax1.set_title(f"{out_col} — Boxplot")
            _style_chart(fig, ax1)

            ax2.hist(col_data, bins=40, color="#3b82f6", alpha=0.75)
            ax2.axvline(lower, color="#ef4444", linestyle="--", linewidth=1.5)
            ax2.axvline(upper, color="#ef4444", linestyle="--", linewidth=1.5)
            ax2.set_title(f"{out_col} — Distribution")
            ax2.set_xlabel(out_col); ax2.set_ylabel("Frequency")
            _style_chart(fig, ax2)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            if count == 0:
                st.success(f"No outliers in {out_col}.")
            else:
                st.warning(f"{count} outlier row(s) in {out_col}.")
                action = st.radio("Action",
                    ["Remove rows","Cap to bounds","Replace with NaN"],
                    horizontal=True, key="out_action")
                if st.button(f"Apply to {out_col}", key="out_apply"):
                    tmp = working_df.copy()
                    try:
                        m2 = (tmp[out_col]<lower)|(tmp[out_col]>upper)
                        if action == "Remove rows":
                            tmp = tmp[~m2]; st.success(f"Removed {count} rows.")
                        elif action == "Cap to bounds":
                            tmp[out_col] = tmp[out_col].clip(lower=lower,upper=upper)
                            st.success(f"Values capped to [{lower:.2f}, {upper:.2f}].")
                        else:
                            tmp.loc[m2, out_col] = np.nan
                            st.success(f"{count} outliers replaced with NaN.")
                        st.session_state.cleaned_df = tmp; st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

    # ── Final preview ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Cleaned Dataset Preview")
    final = st.session_state.cleaned_df
    fc1,fc2,fc3 = st.columns(3)
    fc1.metric("Rows",    f"{final.shape[0]:,}")
    fc2.metric("Columns", final.shape[1])
    fc3.metric("Missing", f"{round(final.isnull().sum().sum()/(final.shape[0]*final.shape[1])*100,1)}%")
    st.dataframe(final.head(10), use_container_width=True)

    csv_out = io.StringIO(); final.to_csv(csv_out, index=False)
    st.download_button(
        label="Download Cleaned CSV", data=csv_out.getvalue(),
        file_name=f"cleaned_{st.session_state.dataset_name}",
        mime="text/csv", key="download_final"
    )
