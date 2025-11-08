import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure Streamlit
st.set_page_config(page_title="Wellbeing & Lifestyle EDA", layout="wide")
st.title("📊 Wellbeing & Lifestyle Data Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    # Basic info
    st.write("### Dataset Information")
    st.write("Shape:", df.shape)
    st.write("Column types:")
    st.write(df.dtypes)

    # Missing values summary
    st.write("### Missing Values per Column")
    st.write(df.isna().sum())

    # Convert Timestamp to datetime
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Convert AGE & GENDER if they exist
    for col in ["AGE", "GENDER"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # ---- Numeric Distributions ----
    st.write("### Numeric Column Distributions")
    cols = st.columns(2)
    for i, col in enumerate(numeric_cols):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            ax.set_title(f"{col}", fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            st.pyplot(fig, use_container_width=True)

    # ---- Categorical Counts ----
    categorical_cols = [c for c in ["AGE", "GENDER"] if c in df.columns]
    if categorical_cols:
        st.write("### Categorical Column Counts")
        cols = st.columns(2)
        for i, col in enumerate(categorical_cols):
            with cols[i % 2]:
                fig, ax = plt.subplots(figsize=(3, 2))
                sns.countplot(x=col, data=df, ax=ax)
                ax.set_title(f"{col}", fontsize=10)
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                st.pyplot(fig, use_container_width=True)

    # ---- Remove Outliers ----
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # ---- Correlation Heatmap ----
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(50, 35))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
    st.pyplot(fig, use_container_width=True)

    # ---- Jointplot ----
    if {"SLEEP_HOURS", "DAILY_STRESS"}.issubset(df.columns):
        st.write("### Sleep Hours vs Daily Stress")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.scatterplot(x="SLEEP_HOURS", y="DAILY_STRESS", data=df, alpha=0.6, ax=ax)
        st.pyplot(fig, use_container_width=True)

    # ---- Boxplots ----
    if {"AGE", "SLEEP_HOURS"}.issubset(df.columns):
        st.write("### Sleep Hours by Age Group")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(x="AGE", y="SLEEP_HOURS", data=df, ax=ax)
        st.pyplot(fig, use_container_width=True)

    if {"GENDER", "DAILY_STRESS"}.issubset(df.columns):
        st.write("### Daily Stress by Gender")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(x="GENDER", y="DAILY_STRESS", data=df, ax=ax)
        st.pyplot(fig, use_container_width=True)

else:
    st.info("👆 Upload a CSV file to begin the analysis.")
