# YouTube Data Analysis and Visualization

## Overview
This project is a **Streamlit-based web application** that allows users to fetch YouTube video data based on a search query, store it in an SQLite database, and analyze it using various techniques such as:
- **Exploratory Data Analysis (EDA)**
- **Time Series Analysis** (using Prophet and moving averages)
- **Sentiment Analysis** (on video comments)

## Features
- **Fetch YouTube video data** using a query.
- **EDA**: Summary statistics, word clouds, and visual insights.
- **Time Series Analysis**: Forecasting YouTube video views.
- **Sentiment Analysis**: Analyzing comment sentiments.
- **Streamlit UI** for easy navigation and interactive visualization.

## Folder Structure
```
├── scripts
│   ├── data_collection.py       # Fetch YouTube video data and store it in SQLite
│   ├── eda.py                   # Perform Exploratory Data Analysis
│   ├── timeSeries.py            # Conduct Time Series Analysis (Prophet, ACF/PACF)
│   ├── sentimentAnalysis.py     # Perform Sentiment Analysis on video comments
├── app.py                        # Main Streamlit App
├── youtube_data.db               # SQLite database storing fetched YouTube data
```

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed along with **pip**.

### Install Dependencies
Run the following command to install required packages:
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

## Usage
1. **Search & Fetch Data**
   - Enter a search query in the input field.
   - Specify the number of videos to retrieve.
   - Click **Fetch Data** to store results in SQLite.

2. **Select Analysis Type**
   - Choose from **EDA, Time Series, or Sentiment Analysis**.
   - Click **Run Analysis** to generate insights.

## Functionality Details
### Objective
- Analyze YouTube video performance metrics (views, likes, comments, etc.).
- Identify trends in content engagement, topics, and audience behavior.
- Visualize insights using Python, Pandas, Matplotlib, and Seaborn.
- Predict video popularity using machine learning (optional).

### Data Collection (YouTube API Integration)
- Setup API Key and authenticate requests.
- Fetch Data: Extract video stats (views, likes, comments, publish date, etc.).
- Scrape Comments for sentiment analysis.
- Store Data in CSV/SQLite/PostgreSQL.

### Data Processing & Cleaning
- Handle missing values (e.g., videos with disabled likes/comments).
- Convert timestamps to datetime for time series analysis.
- Normalize data (log-transform views, likes for better visualization).
- Extract keywords from video titles/descriptions for trend analysis.

### Exploratory Data Analysis (EDA)
- Descriptive Statistics (average views, engagement ratios, top categories).
- Correlation Analysis (Do likes and comments correlate with views?).
- Engagement Ratios: Likes-to-views, comments-to-views distribution.
- **Visualizations:**
  ✅ Bar charts (top-performing videos by views)
  ✅ Scatter plots (likes vs. views, comments vs. views)
  ✅ Histograms (view distribution, engagement per category)

### Time Series Analysis (Trend Detection)
- Analyze view trends over time (growth patterns, seasonal spikes).
- Moving Averages & Forecasting: Predict future video engagement.
- **Visualizations:**
  ✅ Line plots (view growth over time)
  ✅ Seasonal decomposition plots

### Sentiment Analysis on Comments
- Use VADER/TextBlob to classify comments as positive, neutral, or negative.
- Analyze sentiment trends across different types of videos.
- Identify controversial topics with high polarity variance.
- **Visualizations:**
  ✅ Pie charts (sentiment distribution)
  ✅ Box plots (sentiment score per category)

### Topic Modeling (Trending Topics Analysis)
- Identify rising and declining content categories.

### Dashboard for Interactive Insights [Bonus]
- Build a Streamlit/Plotly Dash app for live data visualization.
- Allow users to filter data by category, time range, engagement level.

## Technologies Used
- **Python**
- **Streamlit** (for UI & visualization)
- **SQLite** (for database storage)
- **Prophet** (for time series forecasting)
- **Matplotlib & Seaborn** (for data visualization)
- **NLTK** (for sentiment analysis)


