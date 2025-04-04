import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

DB_NAME = "youtube_data.db"

def load_data():
    """Load data from SQLite into a Pandas DataFrame"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM videos", conn)
    conn.close()
    return df

def show_eda():
    """Streamlit EDA Analysis"""

    st.title("📊 Exploratory Data Analysis")

    # Load Data
    df = load_data()
    st.write("### Dataset Overview")
    st.dataframe(df)

    # Summary Statistics
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Key Metrics
    st.write("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Avg Views", f"{df['views'].mean():,.0f}")
    col2.metric("👍 Avg Likes", f"{df['likes'].mean():,.0f}")
    col3.metric("💬 Avg Comments", f"{df['comments'].mean():,.0f}")
    col4.metric("⏳ Avg Duration (s)", f"{df['duration'].mean():.2f}")

    # Most Popular Videos
    st.write("### Most Popular Videos")
    col1, col2, col3 = st.columns(3)

    most_viewed = df.loc[df["views"].idxmax(), ["title", "views"]]
    col1.write(f"🔥 **Most Viewed:** {most_viewed['title']} ({most_viewed['views']:,.0f} views)")

    most_liked = df.loc[df["likes"].idxmax(), ["title", "likes"]]
    col2.write(f"❤️ **Most Liked:** {most_liked['title']} ({most_liked['likes']:,.0f} likes)")

    most_commented = df.loc[df["comments"].idxmax(), ["title", "comments"]]
    col3.write(f"💬 **Most Commented:** {most_commented['title']} ({most_commented['comments']:,.0f} comments)")

    # Histogram Analysis
    st.write("### Data Distribution")
    column_choice = st.selectbox("Select a column for histogram analysis", ["views", "likes", "comments", "duration"])
    interpret_histogram(df[column_choice], column_choice)

    # Correlation Heatmap
    st.write("### Correlation Matrix")
    interpret_correlation(df)

def interpret_histogram(data, column_name):
    """Plot Histogram and show insights in Streamlit"""
    st.write(f"#### Distribution of {column_name}")

    plt.figure(figsize=(10, 5))
    sns.histplot(data, bins=20, kde=True, color="blue", alpha=0.6)
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column_name}")
    st.pyplot(plt)

    mean_value = np.mean(data)
    median_value = np.median(data)
    std_dev = np.std(data)
    min_value, max_value = np.min(data), np.max(data)

    skewness = "Right-Skewed" if mean_value > median_value else "Left-Skewed" if mean_value < median_value else "Symmetrical"

    variance_type = "High Variance (Widely spread)" if std_dev > mean_value else "Low Variance (Concentrated)"

    threshold = mean_value + 3 * std_dev
    outlier_count = sum(data > threshold)

    # Show insights
    st.markdown(f"""
    **Histogram Insights for {column_name}:**
    - The data is **{skewness}**, meaning most values are {'low' if skewness == 'Right-Skewed' else 'high'}.
    - **Mean:** {mean_value:.2f}, **Median:** {median_value:.2f}
    - **Spread:** {variance_type}
    - **Min:** {min_value}, **Max:** {max_value}
    - **Outliers:** {outlier_count} extreme values above {threshold:.2f}
    """)

def interpret_correlation(df):
    """Show correlation matrix in Streamlit"""
    correlation_matrix = df[["views", "likes", "comments", "subscribers"]].corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    st.pyplot(plt)

    # Generate text-based insights
    interpretation = []
    processed_pairs = set()

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2 and (col2, col1) not in processed_pairs:
                corr_value = correlation_matrix.loc[col1, col2]
                processed_pairs.add((col1, col2))

                if abs(corr_value) >= 0.8:
                    relation = "strong positive" if corr_value > 0 else "strong negative"
                elif abs(corr_value) >= 0.5:
                    relation = "moderate positive" if corr_value > 0 else "moderate negative"
                else:
                    relation = "weak or no correlation"

                interpretation.append(f"- **{col1}** and **{col2}** have a **{relation}** ({corr_value:.2f}).")

    st.write("### Correlation Insights")
    st.markdown("\n".join(interpretation))
