import streamlit as st
import subprocess
import os
import json

SCRIPTS_DIR = "scripts"  
DATA_COLLECTION_SCRIPT = os.path.join(SCRIPTS_DIR, "data_collection.py")
EDA_SCRIPT = os.path.join(SCRIPTS_DIR, "eda.py")
TIME_SERIES_SCRIPT = os.path.join(SCRIPTS_DIR, "timeSeries.py")
SENTIMENT_SCRIPT = os.path.join(SCRIPTS_DIR, "sentimentAnalysis.py")

ANALYSIS_SCRIPTS = {
    "EDA": EDA_SCRIPT,
    "Time Series": TIME_SERIES_SCRIPT,
    "Sentiment Analysis": SENTIMENT_SCRIPT
}
st.markdown(
    """
    <style>
        .stTextInput, .stNumberInput, .stButton {
            text-align: center;
        }
        .stApp {
            background: linear-gradient(45deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb, #ffdde1, #d4fc79, #96e6a1, #fbc2eb);
            background-size: 300% 300%;
            animation: gradientBG 12s ease infinite;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=250)
st.title("YouTube Data Analysis")

search_query = st.text_input("Enter Search Query:")
video_count = st.number_input("Enter Number of Videos:", min_value=1, value=10, step=1)

if st.button("Fetch Data"):
    if search_query:
        input_data = json.dumps({"query": search_query, "count": video_count})

        if os.path.exists(DATA_COLLECTION_SCRIPT):
            with st.spinner("Fetching YouTube data..."):
                process = subprocess.Popen(
                    ["python3", DATA_COLLECTION_SCRIPT],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                output, error = process.communicate(input=input_data)

                if error:
                    st.error(f"Error: {error}")
                else:
                    # Extract JSON response from output
                    try:
                        json_output = [line for line in output.split("\n") if line.strip().startswith("{")][-1]
                        response = json.loads(json_output)
                        if response["status"] == "success":
                            st.success(response["message"])
                        else:
                            st.warning(response["message"])
                    except (json.JSONDecodeError, IndexError):
                        st.error("Failed to parse response from `data_collection.py`.")

    # ✅ Analysis Selection
st.subheader("Choose Analysis Type")
selected_analysis = st.selectbox("Select an analysis:", list(ANALYSIS_SCRIPTS.keys()))

# ✅ Run Analysis Button
if st.button("Run Analysis"):
    analysis_script = ANALYSIS_SCRIPTS[selected_analysis]

    if os.path.exists(analysis_script):
        with st.spinner(f"Running {selected_analysis} Analysis..."):
            if selected_analysis == "EDA":
                import eda
                eda.show_eda()
            elif selected_analysis == "Time Series":
                import timeSeries
                timeSeries.show_time_series()
            elif selected_analysis == "Sentiment Analysis":
                import sentimentAnalysis
                sentimentAnalysis.show_sentiment()
    else:
        st.error(f"Error: `{analysis_script}` not found.")
