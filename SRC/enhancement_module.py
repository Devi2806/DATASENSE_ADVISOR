import streamlit as st
import pandas as pd


def analyze_dataset_issues(df):
    results = {
        "missing_values":      {},
        "categorical_columns": [],
        "duplicate_rows":      0,
        "numeric_columns":     [],
    }
    for col, val in df.isnull().sum().items():
        if val > 0:
            results["missing_values"][col] = int(val)
    results["duplicate_rows"]      = int(df.duplicated().sum())
    results["categorical_columns"] = list(df.select_dtypes(include=["object"]).columns)
    results["numeric_columns"]     = list(df.select_dtypes(include=["int64", "float64"]).columns)
    return results


def calculate_readiness_score(df):
    score, issues = 100, []
    if df.isnull().sum().sum() > 0:
        score -= 20; issues.append("Dataset contains missing values")
    if df.duplicated().sum() > 0:
        score -= 10; issues.append("Dataset contains duplicate rows")
    if len(df.select_dtypes(include=["object"]).columns) > 0:
        score -= 10; issues.append("Categorical columns require encoding")
    mc = df.isnull().sum(); mc = mc[mc > 0]
    if len(mc) > 3:
        score -= 10; issues.append("Multiple columns contain missing values")
    return max(score, 0), issues


def show_future_enhancement(df, target_column):
    if target_column not in df.columns:
        target_column = df.columns[-1]

    # ── Theme colours ────────────────────────────────────────────────
    theme = st.session_state.get("theme", "dark")
    if theme == "dark":
        card_bg  = "#161d2e"
        card_br  = "#1e293b"
        bar_bg   = "#1e293b"
        txt      = "#f1f5f9"
        muted    = "#94a3b8"
        acc      = "#00e5a0"
        red_br   = "rgba(239,68,68,.4)"
        red_txt  = "#f87171"
        red_code = "#fca5a5"
        yel_br   = "rgba(245,158,11,.4)"
        yel_txt  = "#fbbf24"
        yel_code = "#fde68a"
        tip_txt  = "#6ee7b7"
        val_txt  = "#f1f5f9"
    else:
        card_bg  = "#ffffff"
        card_br  = "#e2e8f0"
        bar_bg   = "#e2e8f0"
        txt      = "#0f172a"
        muted    = "#64748b"
        acc      = "#059669"
        red_br   = "rgba(239,68,68,.3)"
        red_txt  = "#dc2626"
        red_code = "#b91c1c"
        yel_br   = "rgba(245,158,11,.3)"
        yel_txt  = "#d97706"
        yel_code = "#b45309"
        tip_txt  = "#059669"
        val_txt  = "#0f172a"

    st.header("Dataset Improvement Suggestions")
    st.markdown(
        f"<p style='color:{muted};margin-top:-0.3rem;'>"
        "Identifies dataset issues and provides actionable recommendations "
        "to prepare your dataset for model training.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    score, issues = calculate_readiness_score(df)

    # ── Readiness score card ─────────────────────────────────────────
    st.subheader("Dataset Readiness Score")
    score_color = acc if score >= 85 else "#f59e0b" if score >= 70 else "#ef4444"

    st.markdown(
        f'<div style="background:{card_bg};border:1px solid {card_br};'
        f'border-radius:12px;padding:1.5rem;margin-bottom:1rem;">'
        f'<p style="color:{muted};font-size:0.72rem;margin:0 0 0.4rem 0;'
        f'text-transform:uppercase;letter-spacing:1px;">Dataset Quality Score</p>'
        f'<p style="color:{score_color};font-size:2.6rem;font-weight:700;margin:0;'
        f'font-family:JetBrains Mono,monospace;">'
        f'{score}<span style="font-size:1.2rem;color:{muted};"> / 100</span></p>'
        f'<div style="background:{bar_bg};border-radius:999px;height:8px;margin-top:1rem;">'
        f'<div style="background:{score_color};width:{score}%;height:8px;border-radius:999px;"></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    if score >= 85:
        st.success("Dataset is ready for model training.")
    elif score >= 70:
        st.warning("Dataset is almost ready but requires minor preprocessing.")
    else:
        st.error("Dataset needs significant preprocessing before training.")

    if issues:
        st.markdown(f"<p style='color:{txt};font-weight:600;margin-top:0.5rem;'>Issues affecting score:</p>",
                    unsafe_allow_html=True)
        for issue in issues:
            st.markdown(f"<p style='color:{muted};margin:2px 0;'>— {issue}</p>", unsafe_allow_html=True)

    st.markdown("---")
    analysis = analyze_dataset_issues(df)

    # ── Missing values ────────────────────────────────────────────────
    st.subheader("Missing Value Analysis")

    if not analysis["missing_values"]:
        st.success("No missing values detected.")
    else:
        for col, val in analysis["missing_values"].items():
            st.markdown(
                f'<div style="background:{card_bg};border:1px solid {red_br};'
                f'border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:0.6rem;">'
                f'<p style="color:{red_txt};font-weight:600;margin:0 0 0.25rem 0;">'
                f'Column: <code style="color:{red_code};background:transparent;">{col}</code></p>'
                f'<p style="color:{muted};margin:0 0 0.2rem 0;">'
                f'Missing Count: <strong style="color:{val_txt};">{val}</strong></p>'
                f'<p style="color:{tip_txt};margin:0;">'
                f'Apply imputation (mean / median / mode) or drop rows.</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Duplicate rows ────────────────────────────────────────────────
    st.subheader("Duplicate Data Check")

    if analysis["duplicate_rows"] == 0:
        st.success("No duplicate rows detected.")
    else:
        st.error(f"{analysis['duplicate_rows']} duplicate rows found.")
        st.code("df = df.drop_duplicates()", language="python")

    st.markdown("---")

    # ── Categorical encoding ──────────────────────────────────────────
    st.subheader("Categorical Features")

    if not analysis["categorical_columns"]:
        st.success("No categorical columns — no encoding needed.")
    else:
        for col in analysis["categorical_columns"]:
            st.markdown(
                f'<div style="background:{card_bg};border:1px solid {yel_br};'
                f'border-radius:8px;padding:0.85rem 1.1rem;margin-bottom:0.6rem;">'
                f'<p style="color:{yel_txt};font-weight:600;margin:0 0 0.25rem 0;">'
                f'Column: <code style="color:{yel_code};background:transparent;">{col}</code></p>'
                f'<p style="color:{muted};margin:0 0 0.2rem 0;">ML models require numerical inputs.</p>'
                f'<p style="color:{tip_txt};margin:0;">Apply Label Encoding or One-Hot Encoding.</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Feature scaling ───────────────────────────────────────────────
    st.subheader("Feature Scaling Recommendation")

    if len(analysis["numeric_columns"]) > 1:
        cols_str = ", ".join(analysis["numeric_columns"])
        st.info(f"Numeric columns detected: {cols_str}")
        st.success("Apply StandardScaler or MinMaxScaler before training.")
    else:
        st.info("Only one numeric column — scaling may not be necessary.")
