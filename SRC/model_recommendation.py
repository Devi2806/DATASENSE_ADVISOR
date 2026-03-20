import streamlit as st
import pandas as pd


def detect_problem_type(df, target_column):
    if df[target_column].dtype == "object":
        return "classification"
    if df[target_column].nunique() < 10:
        return "classification"
    return "regression"


def analyze_dataset(df):
    insights = []
    rows, cols = df.shape
    if rows < 1000:
        insights.append("Small dataset — Simple models like Logistic Regression or SVM may work well.")
    elif rows < 10000:
        insights.append("Medium dataset — Tree-based models like Random Forest can perform well.")
    else:
        insights.append("Large dataset — Ensemble models like Gradient Boosting or Random Forest are recommended.")
    missing_percent = (df.isnull().sum().sum() / (rows * cols)) * 100
    if missing_percent > 10:
        insights.append("High missing values — Tree-based models handle missing data better.")
    cat_ratio = len(df.select_dtypes(include=["object"]).columns) / cols
    if cat_ratio > 0.4:
        insights.append("Many categorical features — Decision Trees and Random Forest are preferred.")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        if (corr_matrix > 0.8).sum().sum() > numeric_df.shape[1]:
            insights.append("High feature correlation — Linear models may struggle. Tree models recommended.")
    return insights


def select_top_models(df, problem_type):
    rows, cols = df.shape
    cat_ratio = len(df.select_dtypes(include=["object"]).columns) / cols
    missing_percent = (df.isnull().sum().sum()) / (rows * cols) * 100

    if problem_type == "classification":
        if rows < 1000:
            return [
                {"model": "Logistic Regression",       "reason": "Small dataset — works well for simpler classification problems."},
                {"model": "Support Vector Machine",     "reason": "SVM performs well on smaller datasets with clear class separation."},
                {"model": "K-Nearest Neighbors (KNN)", "reason": "KNN is effective for small datasets and simple classification tasks."},
            ]
        elif cat_ratio > 0.4:
            return [
                {"model": "Decision Tree",     "reason": "Many categorical features — Decision Trees handle them naturally."},
                {"model": "Random Forest",     "reason": "Random Forest improves Decision Tree performance via ensemble learning."},
                {"model": "Gradient Boosting", "reason": "Boosting models perform well on structured datasets with categorical variables."},
            ]
        elif missing_percent > 10:
            return [
                {"model": "Random Forest", "reason": "High missing values — Random Forest handles missing data better."},
                {"model": "Decision Tree", "reason": "Decision Trees can split data even with missing values."},
                {"model": "XGBoost",       "reason": "XGBoost is robust and performs well with incomplete datasets."},
            ]
        else:
            return [
                {"model": "Random Forest",     "reason": "Balanced dataset — Random Forest provides strong baseline performance."},
                {"model": "Gradient Boosting", "reason": "Boosting methods often achieve high accuracy on structured datasets."},
                {"model": "XGBoost",           "reason": "XGBoost is one of the most powerful ensemble algorithms."},
            ]
    else:
        if rows < 1000:
            return [
                {"model": "Linear Regression",        "reason": "Small dataset — provides a simple interpretable baseline."},
                {"model": "Ridge Regression",         "reason": "Ridge helps control overfitting in smaller datasets."},
                {"model": "Support Vector Regressor", "reason": "SVR can capture complex relationships in smaller datasets."},
            ]
        elif missing_percent > 10:
            return [
                {"model": "Random Forest Regressor",     "reason": "Tree-based models handle missing values more effectively."},
                {"model": "Decision Tree Regressor",     "reason": "Decision Trees capture nonlinear patterns even with incomplete data."},
                {"model": "Gradient Boosting Regressor", "reason": "Boosting improves prediction accuracy on structured regression datasets."},
            ]
        else:
            return [
                {"model": "Random Forest Regressor",     "reason": "Strong baseline performance for regression tasks."},
                {"model": "Gradient Boosting Regressor", "reason": "Boosting algorithms often outperform simple regression models."},
                {"model": "XGBoost Regressor",           "reason": "XGBoost is widely used for high-performance regression tasks."},
            ]


def recommend_models(problem_type):
    if problem_type == "classification":
        return [
            {"model": "Logistic Regression",       "accuracy": "70% - 80%", "reason": "Baseline algorithm for linear classification problems."},
            {"model": "K-Nearest Neighbors (KNN)", "accuracy": "65% - 80%", "reason": "Simple distance-based algorithm for smaller datasets."},
            {"model": "Naive Bayes",               "accuracy": "65% - 80%", "reason": "Works well for text and probabilistic classification."},
            {"model": "Decision Tree",             "accuracy": "70% - 85%", "reason": "Handles nonlinear relationships and mixed data types."},
            {"model": "Random Forest",             "accuracy": "80% - 90%", "reason": "Ensemble model that reduces overfitting."},
            {"model": "Support Vector Machine",    "accuracy": "75% - 90%", "reason": "Effective for high-dimensional feature spaces."},
            {"model": "Gradient Boosting",         "accuracy": "80% - 92%", "reason": "Boosting technique that improves weak learners sequentially."},
            {"model": "XGBoost",                   "accuracy": "85% - 95%", "reason": "Highly optimised gradient boosting, popular in competitions."},
        ]
    else:
        return [
            {"model": "Linear Regression",           "accuracy": "60% - 75%", "reason": "Simple baseline for linear relationships."},
            {"model": "Ridge Regression",            "accuracy": "65% - 80%", "reason": "Regularised regression that reduces overfitting."},
            {"model": "Lasso Regression",            "accuracy": "65% - 80%", "reason": "Performs feature selection by shrinking coefficients."},
            {"model": "Decision Tree Regressor",     "accuracy": "65% - 80%", "reason": "Captures nonlinear patterns in regression tasks."},
            {"model": "Random Forest Regressor",     "accuracy": "75% - 90%", "reason": "Ensemble model with strong generalisation."},
            {"model": "Support Vector Regressor",    "accuracy": "70% - 85%", "reason": "Effective for complex regression relationships."},
            {"model": "Gradient Boosting Regressor", "accuracy": "80% - 92%", "reason": "Boosting algorithm that improves predictive performance."},
            {"model": "XGBoost Regressor",           "accuracy": "85% - 95%", "reason": "Highly optimised boosting model."},
        ]


def show_model_recommendation(df, target_column):
    if target_column not in df.columns:
        target_column = df.columns[-1]

    # Read theme for card colours
    theme   = st.session_state.get("theme", "dark")
    card_bg = "#161d2e"   if theme == "dark" else "#ffffff"
    card_br = "#1e293b"   if theme == "dark" else "#e2e8f0"
    acc     = "#00e5a0"   if theme == "dark" else "#059669"
    txt     = "#f1f5f9"   if theme == "dark" else "#0f172a"
    muted   = "#94a3b8"   if theme == "dark" else "#64748b"

    st.header("Model Recommendation")
    st.markdown(
        f"<p style='color:{muted};margin-top:-0.3rem;'>"
        "AI-powered model suggestions based on your dataset characteristics.</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    problem_type  = detect_problem_type(df, target_column)
    dataset_size  = df.shape[0]
    feature_count = df.shape[1] - 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",         f"{dataset_size:,}")
    c2.metric("Features",     feature_count)
    c3.metric("Problem Type", problem_type.capitalize())

    st.markdown("---")

    # AI insights
    st.subheader("AI Dataset Insights")
    insights = analyze_dataset(df)
    if not insights:
        st.success("Dataset looks clean and well-structured.")
    else:
        for insight in insights:
            st.info(insight)

    st.markdown("---")

    # Top 3 models — theme-aware cards
    st.subheader("Top 3 Recommended Models")
    st.caption("Selected based on your dataset characteristics")

    top_models = select_top_models(df, problem_type)
    for i, model in enumerate(top_models, start=1):
        st.markdown(
            f'<div style="background:{card_bg};border:1px solid {card_br};'
            f'border-left:4px solid {acc};border-radius:10px;'
            f'padding:1rem 1.2rem;margin-bottom:0.6rem;">'
            f'<p style="color:{acc};font-weight:700;font-size:0.95rem;margin:0 0 0.3rem 0;">'
            f'#{i} &nbsp; {model["model"]}</p>'
            f'<p style="color:{muted};margin:0;font-size:0.85rem;">{model["reason"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # All models
    st.subheader("All Possible Models")
    st.caption("Click any model to expand details")
    models = recommend_models(problem_type)
    for m in models:
        with st.expander(m["model"]):
            st.markdown(f"**Expected Accuracy:** `{m['accuracy']}`")
            st.markdown(f"**Why this model:** {m['reason']}")
            st.info("Actual performance depends on dataset quality, feature engineering, and hyperparameter tuning.")
