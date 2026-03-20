import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------
# DARK THEME HELPER FOR MATPLOTLIB
# BUG FIX: All charts now match the dark UI instead of rendering white
# ------------------------------
def apply_dark_style(fig, ax):
    """Apply consistent dark theme to all matplotlib figures."""
    bg = "#161d2e"
    fg = "#f1f5f9"
    grid = "#1e293b"
    accent = "#00e5a0"

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.spines["bottom"].set_color(grid)
    ax.spines["left"].set_color(grid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(colors=fg, labelsize=9)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(accent)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")

    ax.yaxis.set_tick_params(labelcolor=fg)
    ax.xaxis.set_tick_params(labelcolor=fg)


# -------------------------
# BASIC DATASET OVERVIEW
# -------------------------
def basic_overview(df):

    st.subheader("📋 Data Types Overview")

    dtype_df = df.dtypes.reset_index()
    dtype_df.columns = ["Column Name", "Data Type"]
    dtype_df["Data Type"] = dtype_df["Data Type"].astype(str)

    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # -------------------------
    # Missing Values
    # -------------------------
    st.subheader("⚠️ Missing Values")

    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column Name", "Missing Count"]
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        st.success("✅ No missing values found in this dataset.")
    else:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

        # BUG FIX: Added plt.close(fig) to avoid memory leak
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            missing_df["Column Name"],
            missing_df["Missing Count"],
            color="#00e5a0",
            alpha=0.85,
            width=0.6
        )
        ax.set_title("Missing Values per Column")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")
        apply_dark_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------
    # Duplicate Rows
    # -------------------------
    st.subheader("🔁 Duplicate Rows")

    duplicates = int(df.duplicated().sum())
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Total Duplicates", duplicates)
    with col2:
        if duplicates == 0:
            st.success("✅ No duplicate rows detected.")
        else:
            st.warning(f"⚠️ {duplicates} duplicate rows found. Consider using `df.drop_duplicates()`.")

    # -------------------------
    # Statistical Summary
    # -------------------------
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe().round(4), use_container_width=True)


# -------------------------
# FEATURE ANALYSIS
# -------------------------
def feature_analysis(df):

    st.subheader("📈 Numeric Feature Distribution")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(num_cols) == 0:
        st.warning("No numeric columns found in this dataset.")
    else:
        for col in num_cols[:4]:  # Limit to 4 for clean UI
            col1, col2 = st.columns(2)

            with col1:
                # BUG FIX: Added plt.close(fig) after each figure
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(df[col].dropna(), bins=30, color="#3b82f6", alpha=0.85, edgecolor="#111827")
                ax.set_title(f"{col} — Distribution")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                apply_dark_style(fig, ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                bp = ax.boxplot(
                    df[col].dropna(),
                    patch_artist=True,
                    boxprops=dict(facecolor="#1e293b", color="#00e5a0"),
                    medianprops=dict(color="#00e5a0", linewidth=2),
                    whiskerprops=dict(color="#94a3b8"),
                    capprops=dict(color="#94a3b8"),
                    flierprops=dict(marker="o", color="#f59e0b", markersize=4)
                )
                ax.set_title(f"{col} — Boxplot")
                apply_dark_style(fig, ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # -------------------------
    # Categorical Analysis
    # -------------------------
    st.subheader("📊 Categorical Feature Distribution")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_cols) == 0:
        st.warning("No categorical columns found in this dataset.")
    else:
        # BUG FIX: Correct rename — reset_index() on a Series creates columns "index" and 0
        unique_counts = df[cat_cols].nunique().reset_index()
        unique_counts.columns = ["Column Name", "Unique Count"]
        st.dataframe(unique_counts, use_container_width=True, hide_index=True)

        for col in cat_cols:
            unique_count = df[col].nunique()

            if unique_count > 20:
                st.info(f"Skipping **{col}** — too many categories ({unique_count})")
                continue

            top_vals = df[col].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.bar(top_vals.index.astype(str), top_vals.values, color="#8b5cf6", alpha=0.85, width=0.6)
            ax.set_title(f"{col} — Top Categories")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            apply_dark_style(fig, ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# -------------------------
# ADVANCED ANALYSIS
# -------------------------
def advanced_analysis(df, target_column):

    st.subheader("🔬 Advanced Analysis")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # -------------------------
    # Correlation Heatmap
    # -------------------------
    if numeric_df.shape[1] > 1:

        st.subheader("🔗 Correlation Heatmap")

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) - 1)))

        # BUG FIX: Use imshow instead of matshow for better dark theme integration
        cax = ax.imshow(corr, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)

        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#f1f5f9")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#f1f5f9")

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)

        # Annotate cells with correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.5 else "#94a3b8")

        apply_dark_style(fig, ax)
        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Not enough numeric features for a correlation heatmap.")

    # -------------------------
    # Target Variable Analysis
    # -------------------------
    st.subheader("🎯 Target Variable Analysis")

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataset.")
        return

    if df[target_column].dtype == "object" or df[target_column].nunique() < 20:
        st.success("📊 **Classification Problem Detected**")

        value_counts = df[target_column].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(value_counts.index.astype(str), value_counts.values, color="#00e5a0", alpha=0.85, width=0.6)
        ax.set_title(f"{target_column} — Class Distribution")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        apply_dark_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.success("📈 **Regression Problem Detected**")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[target_column].dropna(), bins=30, color="#3b82f6", alpha=0.85, edgecolor="#111827")
        ax.set_title(f"{target_column} — Distribution")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Frequency")
        apply_dark_style(fig, ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
