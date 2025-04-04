import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ✅ Streamlit Function for Sentiment Analysis
def show_sentiment():
    st.title("📊 YouTube Video Sentiment Analysis")

    # ✅ Load Data from SQLite
    @st.cache_data
    def load_videos():
        conn = sqlite3.connect("youtube_data.db")
        query = "SELECT video_id, title, views, channel_name, subscribers FROM videos"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            st.warning("No video data found in the database.")
            return None

        df["channel_name"] = df["channel_name"].fillna("Unknown")
        return df

    df = load_videos()
    if df is None:
        return

    # ✅ Sentiment Analysis Functions
    vader_analyzer = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        score = vader_analyzer.polarity_scores(text)["compound"]
        return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"

    def get_textblob_sentiment(text):
        score = TextBlob(text).sentiment.polarity
        return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

    # ✅ Perform Sentiment Analysis
    df["VADER Sentiment"] = df["title"].fillna("").apply(get_vader_sentiment)
    df["TextBlob Sentiment"] = df["title"].fillna("").apply(get_textblob_sentiment)

    # Combine results (if both agree, keep that; otherwise, mark as Neutral)
    df["Final Sentiment"] = df.apply(
        lambda row: row["VADER Sentiment"] if row["VADER Sentiment"] == row["TextBlob Sentiment"] else "Neutral",
        axis=1
    )

    # 📊 **1️⃣ Sentiment Distribution Pie Chart**
    def plot_sentiment_distribution():
        sentiment_counts = df["Final Sentiment"].value_counts()
        colors = sns.color_palette("pastel")

        fig, ax = plt.subplots(figsize=(6, 6))
        sentiment_counts.plot.pie(
            autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "black"}, ax=ax
        )
        ax.set_title("Sentiment Distribution of YouTube Videos", fontsize=14, fontweight="bold")
        ax.set_ylabel("")
        st.pyplot(fig)

        st.write("### 🧐 Insights:")
        st.write("- If most videos are **green**, they are positively received.")
        st.write("- A **red majority** means high negative reactions.")
        st.write("- **Gray** suggests neutral or mixed reactions.")

    # 📊 **2️⃣ Sentiment Trends Across Channels (Stacked Bar Chart)**
    def plot_sentiment_by_channel():
        channel_sentiment = df.groupby("channel_name")["Final Sentiment"].value_counts().unstack().fillna(0)
        colors = sns.color_palette("coolwarm", n_colors=3)

        fig, ax = plt.subplots(figsize=(12, 6))
        channel_sentiment.plot(kind="bar", stacked=True, color=colors, edgecolor="black", ax=ax)
        ax.set_xlabel("YouTube Channels", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Videos", fontsize=12, fontweight="bold")
        ax.set_title("Sentiment Trends Across YouTube Channels", fontsize=14, fontweight="bold")
        ax.legend(title="Sentiment", fontsize=10)
        plt.xticks(rotation=45)
        sns.despine()
        st.pyplot(fig)

        st.write("### 🎭 Sentiment Trends:")
        st.write("- High **negative sentiment** suggests controversy.")
        st.write("- A **balanced mix** of colors means varied reception.")

    # 📊 **3️⃣ Sentiment Score Boxplot**
    def plot_sentiment_boxplot():
        df["Sentiment Score"] = df["title"].apply(lambda x: vader_analyzer.polarity_scores(x)["compound"])

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="Final Sentiment", y="Sentiment Score", palette="coolwarm", ax=ax)
        ax.set_title("Sentiment Score Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sentiment Category", fontsize=12)
        ax.set_ylabel("Sentiment Score", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

        st.write("### 🔎 Boxplot Insights:")
        st.write("- Positive and negative sentiments may have **outliers**.")
        st.write("- **Wide distributions** indicate varied sentiment scores.")

    # 📊 **4️⃣ Controversial Channels (High Sentiment Variance)**
    def find_controversial_channels():
        sentiment_variance = df.groupby("channel_name")["Final Sentiment"].value_counts().unstack().var(axis=1).sort_values(ascending=False)

        st.write("### 🔥 Most Controversial Channels:")
        st.write(sentiment_variance.head(5))

    # 📊 **5️⃣ Topic Modeling for Trending Video Topics**
    def topic_modeling():
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        text_data = df["title"].fillna("")
        tfidf_matrix = vectorizer.fit_transform(text_data)

        num_topics = 5
        nmf = NMF(n_components=num_topics, random_state=42)
        nmf.fit(tfidf_matrix)

        feature_names = vectorizer.get_feature_names_out()

        st.write("### 📊 Trending Video Topics:")
        for topic_idx, topic in enumerate(nmf.components_):  # ✅ Fix: Removed ()
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            st.write(f"🔹 **Topic {topic_idx+1}:** {', '.join(top_words)}")


    # ✅ Run All Analysis
    plot_sentiment_distribution()
    plot_sentiment_by_channel()
    plot_sentiment_boxplot()
    find_controversial_channels()
    topic_modeling()
