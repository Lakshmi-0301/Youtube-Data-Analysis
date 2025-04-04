import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Database Name
DB_NAME = "youtube_data.db"

# Function to Load Data
def load_data():
    """Load data from SQLite into a Pandas DataFrame"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM videos", conn)
    conn.close()
    return df

# Function to Display EDA
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

    # **✅ Display Two Histograms Side by Side**
    st.write("### Data Distribution: Views & Likes")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Distribution of Views")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df["views"], bins=20, kde=True, color="blue", alpha=0.6, ax=ax)
        ax.set_xlabel("Views")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Views")
        st.pyplot(fig)

    with col2:
        st.write("#### Distribution of Likes")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(df["likes"], bins=20, kde=True, color="green", alpha=0.6, ax=ax)
        ax.set_xlabel("Likes")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Likes")
        st.pyplot(fig)

    # **Correlation Matrix**
    st.write("### Correlation Matrix")
    interpret_correlation(df)

# Function to Display Correlation Matrix
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

# Run Streamlit App
if __name__ == "__main__":
    show_eda()
