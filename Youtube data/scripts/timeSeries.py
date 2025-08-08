import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

def show_time_series():
    st.title("📈 YouTube Views Time Series Analysis")

    @st.cache_data
    def load_data():
        try:
            conn = sqlite3.connect("youtube_data.db")
            df = pd.read_sql_query("SELECT publish_date, views FROM videos", conn)
            conn.close()
        except Exception as e:
            st.error(f" Database Error: {e}")
            return None

        if "publish_date" not in df.columns or "views" not in df.columns:
            st.error("Error: The database table must contain 'publish_date' and 'views' columns.")
            return None

        if df.empty:
            st.warning("⚠️ No data found in the database.")
            return None

        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df = df.groupby('publish_date').sum().reset_index()
        df = df.rename(columns={'publish_date': 'ds', 'views': 'y'})

        date_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max())
        df = df.set_index('ds').reindex(date_range).reset_index().rename(columns={'index': 'ds'})
        df['y'] = df['y'].fillna(method='ffill')

        return df

    df = load_data()
    if df is None:
        return

    def plot_moving_average():
        df["MA_7"] = df["y"].rolling(window=7, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["ds"], df["y"], label="Actual Views", color="blue", alpha=0.5)
        ax.plot(df["ds"], df["MA_7"], label="7-Day Moving Avg", color="red")
        ax.set_xlabel("Date")
        ax.set_ylabel("Views")
        ax.set_title("YouTube Views with Moving Averages")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        st.write("### 📊 Moving Average Insights:")
        st.write("- The **red line** smooths out short-term fluctuations, revealing overall trends.")
        st.write("- If the moving average is **increasing**, viewership is growing.")

    def forecast_views():
        if len(df) < 10:
            st.warning(" Not enough data for forecasting.")
            return

        prior_scale = 0.05 if len(df) > 200 else 0.1  
        forecast_period = max(7, (df["ds"].max() - df["ds"].min()).days // 3)  

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=prior_scale)
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_period, freq="D")
        forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["ds"], df["y"], label="Actual Views", color="blue")
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Views", color="red", linestyle="dashed")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="pink", alpha=0.3)
        ax.set_xlabel("Date")
        ax.set_ylabel("Views")
        ax.set_title("YouTube View Trends & Forecast")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        st.write("### Forecast Insights:")
        st.write("- **Red dashed line:** Future predicted views.")
        st.write("- **Pink shaded region:** Prediction confidence range.")
        st.write("- **Upward trend** suggests increasing audience engagement.")

        st.write("### 🔍 Trend & Seasonality Breakdown:")
        st.pyplot(model.plot_components(forecast))

    def plot_acf_pacf():
        if len(df) > 15:  
            lags = min(10, len(df) // 2)
            acf_values = acf(df["y"], nlags=lags)
            pacf_values = pacf(df["y"], nlags=lags)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].stem(range(lags+1), acf_values)
            axes[0].set_title("Autocorrelation (ACF) - View Trends")

            axes[1].stem(range(lags+1), pacf_values)
            axes[1].set_title("Partial Autocorrelation (PACF)")

            st.pyplot(fig)

            st.write("### ACF & PACF Insights:")
            st.write("- **ACF:** Measures how past views influence future views. If spikes appear at regular intervals, it suggests repeating cycles (e.g., weekly trends).")
            st.write("- **PACF:** Helps determine the number of lagged values to use in predictive models. Large spikes at specific lags indicate strong direct influence.")

        else:
            st.warning(" Not enough data for ACF/PACF analysis.")

    def plot_time_series():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["ds"], df["y"], color="blue", alpha=0.6)
        ax.set_xlabel("Date")
        ax.set_ylabel("Views")
        ax.set_title("YouTube View Trends Over Time")
        ax.grid()
        st.pyplot(fig)

        st.write("### 📈 Time Series Insights:")
        st.write("- A **rising trend** indicates growing popularity.")
        st.write("- A **declining trend** suggests a drop in viewer interest.")

    plot_moving_average()
    forecast_views()
    plot_time_series()
    plot_acf_pacf()
