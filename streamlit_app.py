import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import openai
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error

# Set Streamlit Page Configurations
st.set_page_config(
    page_title="📊 AI-Powered Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📊 AI-Powered Stock Market Dashboard")
st.markdown("An interactive dashboard for stock price analysis, sentiment trends, technical indicators, and AI-based predictions.")




@st.cache_data
def load_data():
    # Define file paths
    file_paths = {
        "ohlc_df": "ohlc_data_cleaned.csv",
        "sentiment_df": "sentiment_aggregated.csv",
        "technical_df": "technical_indicators_dataset.csv",
        "llm_predictions_df": "llm_quantified_predictions.csv",
        "news_df": "news_articles_cleaned.csv",
    }

    # Load datasets
    dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

    # Convert Date Columns
    date_columns = {
        "ohlc_df": ["DateTime", "Timestamp", "Date"],
        "sentiment_df": ["Date"],
        "technical_df": ["DateTime", "Timestamp", "Date"],
        "llm_predictions_df": ["DateTime", "Timestamp", "Date"],
        "news_df": ["DateTime", "Published Date", "Timestamp"]
    }

    for name, df in dfs.items():
        for col in date_columns[name]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
                df.rename(columns={col: "DateTime"}, inplace=True)  # Standardize column names
                break  # Stop searching after finding the correct column

    # Extract modified dataframes
    ohlc_df = dfs["ohlc_df"]
    sentiment_df = dfs["sentiment_df"]
    technical_df = dfs["technical_df"]
    llm_predictions_df = dfs["llm_predictions_df"]
    news_df = dfs["news_df"]

    return ohlc_df, sentiment_df, technical_df, llm_predictions_df, news_df

# 🚀 Convert Streamlit's `st.date_input()` values to `datetime64[ns]`
def convert_date(date_input):
    """Convert Python `date` object to Pandas `datetime64[ns]`."""
    return pd.to_datetime(datetime.combine(date_input, datetime.min.time()), utc=True)

# **Streamlit Sidebar: User Inputs**
st.sidebar.title("📊 Stock Market Analysis")

# Load datasets
ohlc_df, sentiment_df, technical_df, llm_predictions_df, news_df = load_data()

# **Dropdown to select stock ticker**
available_tickers = ohlc_df["Ticker"].unique()  # Extract unique stock tickers
selected_ticker = st.sidebar.selectbox("Select a Stock Ticker", available_tickers)

# **Date Input for Filtering**
start_date = st.sidebar.date_input("Select Start Date", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("Select End Date", datetime(2025, 1, 1))

# ✅ Convert `st.date_input()` values to `datetime64[ns]`
start_date = convert_date(start_date)
end_date = convert_date(end_date)

# ✅ Ensure `ohlc_df["DateTime"]` is also in `datetime64[ns]`
ohlc_df["DateTime"] = pd.to_datetime(ohlc_df["DateTime"], utc=True).dt.tz_convert(None)

# 🔹 **Check Data Types for Debugging**
print(f"🔹 start_date type: {type(start_date)}, end_date type: {type(end_date)}")
print(f"🔹 ohlc_df['DateTime'] type: {ohlc_df['DateTime'].dtype}")

# **🛠️ Apply Filtering**
ohlc_filtered = ohlc_df[
    (ohlc_df["Ticker"] == selected_ticker) &
    (ohlc_df["DateTime"] >= start_date) &
    (ohlc_df["DateTime"] <= end_date)
]

# **Display Filtered Data**
st.write(f"📊 Showing data for {selected_ticker} from {start_date.date()} to {end_date.date()}")
st.dataframe(ohlc_filtered)



# Sidebar for Stock Selection
st.sidebar.title("📊 Select Stock & Date Range")
selected_ticker = st.sidebar.selectbox("Choose a Stock:", ohlc_df["Ticker"].unique())

# Sidebar Date Range Selection
start_date = st.sidebar.date_input("Start Date", min_value=ohlc_df["DateTime"].min(), max_value=ohlc_df["DateTime"].max(), value=ohlc_df["DateTime"].min())
end_date = st.sidebar.date_input("End Date", min_value=ohlc_df["DateTime"].min(), max_value=ohlc_df["DateTime"].max(), value=ohlc_df["DateTime"].max())

# Filter Data Based on User Selection
ohlc_filtered = ohlc_df[(ohlc_df["Ticker"] == selected_ticker) & (ohlc_df["DateTime"].between(start_date, end_date))]
sentiment_filtered = sentiment_df[(sentiment_df["Ticker"] == selected_ticker) & (sentiment_df["Date"].between(start_date, end_date))]
technical_filtered = technical_df[(technical_df["Ticker"] == selected_ticker) & (technical_df["DateTime"].between(start_date, end_date))]
llm_filtered = llm_predictions_df[(llm_predictions_df["Ticker"] == selected_ticker) & (llm_predictions_df["DateTime"].between(start_date, end_date))]
news_filtered = news_df[(news_df["Ticker"] == selected_ticker) & (news_df["DateTime"].between(start_date, end_date))]


st.subheader(f"📌 Data Preview for {selected_ticker}")
st.write("### 📊 Stock Market Data (Filtered)")
st.dataframe(ohlc_filtered)

st.write("### 📰 Sentiment Data (Filtered)")
st.dataframe(sentiment_filtered)

st.write("### 📈 Technical Indicators (Filtered)")
st.dataframe(technical_filtered)

st.write("### 🤖 AI Predictions (Filtered)")
st.dataframe(llm_filtered)

st.write("### 📰 News Data (Filtered)")
st.dataframe(news_filtered)


# 📊 Stock Price Trend Section
st.subheader(f"📈 Stock Price Trend for {selected_ticker}")

# Plot Closing Price Trends
fig_price_trend = px.line(
    ohlc_filtered,
    x="DateTime",
    y="Close",
    title=f"Closing Price Trend for {selected_ticker}",
    labels={"Close": "Closing Price ($)", "DateTime": "Date"},
    markers=True
)
st.plotly_chart(fig_price_trend, use_container_width=True)



# 📰 Sentiment Analysis Section
st.subheader("📢 Sentiment Analysis")

# Display Sentiment Trends
fig_sentiment = px.line(
    sentiment_filtered,
    x="Date",
    y="Sentiment_Score_Avg",
    title=f"📊 Sentiment Score Trend for {selected_ticker}",
    labels={"Sentiment_Score_Avg": "Sentiment Score", "Date": "Date"},
    markers=True
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# Display Recent News Articles
st.write("### 📰 Latest News")
for index, row in news_filtered.head(5).iterrows():
    st.markdown(f"**{row['Title']}**")
    st.markdown(f"*Published on: {row['DateTime'].date()}*")
    st.markdown(f"[Read more]({row['URL']})")
    st.write("---")



# 📈 Technical Indicators Section
st.subheader("📊 Technical Indicators")

# Sidebar Multi-Select for Technical Indicators
indicator_options = ["RSI", "MACD", "Volatility", "Momentum"]
selected_indicators = st.sidebar.multiselect("Select Technical Indicators", indicator_options, default=["RSI", "MACD"])

# Plot Technical Indicators
fig_tech = px.line(
    technical_filtered,
    x="DateTime",
    y=selected_indicators,
    title=f"📊 Selected Technical Indicators for {selected_ticker}",
    labels={"value": "Indicator Value", "DateTime": "Date"},
    markers=True
)
st.plotly_chart(fig_tech, use_container_width=True)


# 🔎 AI Predictions vs. Market Trends
st.subheader("🤖 AI Predictions vs. Market Trends")

# Merge LLM Predictions with OHLC Data
llm_vs_market_df = llm_filtered.merge(
    ohlc_filtered[["DateTime", "Ticker", "Close"]],
    on=["DateTime", "Ticker"],
    how="left"
)

# Plot LLM Predictions vs. Actual Market Movement
fig_llm = px.line(
    llm_vs_market_df,
    x="DateTime",
    y=["LLM_Prediction", "Close"],
    title=f"📈 AI Predictions vs. Actual Stock Price for {selected_ticker}",
    labels={"value": "Stock Price / Prediction", "DateTime": "Date"},
    markers=True
)
st.plotly_chart(fig_llm, use_container_width=True)



# 🎯 Model Performance Evaluation
st.subheader("🎯 AI Model Performance Metrics")

# Compute Model Evaluation Metrics
accuracy = accuracy_score(llm_filtered["Actual_Trend"], llm_filtered["LLM_Prediction"]) * 100
precision = precision_score(llm_filtered["Actual_Trend"], llm_filtered["LLM_Prediction"], average="weighted", zero_division=0)
recall = recall_score(llm_filtered["Actual_Trend"], llm_filtered["LLM_Prediction"], average="weighted", zero_division=0)
rmse = mean_squared_error(llm_filtered["Actual_Trend"], llm_filtered["LLM_Prediction"], squared=False)

# Display Metrics
st.write(f"📌 **Prediction Accuracy:** {accuracy:.2f}%")
st.write(f"📍 **Precision:** {precision:.2f}")
st.write(f"📍 **Recall:** {recall:.2f}")
st.write(f"📍 **Root Mean Squared Error (RMSE):** {rmse:.2f}")


# Optimize dataset loading with caching
@st.cache_data
def load_data():
    ohlc_df = pd.read_csv("ohlc_data_cleaned.csv")
    sentiment_df = pd.read_csv("sentiment_aggregated.csv")
    technical_df = pd.read_csv("technical_indicators_dataset.csv")
    llm_predictions_df = pd.read_csv("llm_quantified_predictions.csv")
    news_df = pd.read_csv("news_articles_cleaned.csv")

    # Convert Date Columns
    ohlc_df["DateTime"] = pd.to_datetime(ohlc_df["DateTime"], utc=True).dt.tz_convert(None)
    sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
    technical_df["DateTime"] = pd.to_datetime(technical_df["DateTime"], utc=True).dt.tz_convert(None)
    llm_predictions_df["DateTime"] = pd.to_datetime(llm_predictions_df["DateTime"], utc=True).dt.tz_convert(None)
    news_df["DateTime"] = pd.to_datetime(news_df["DateTime"], utc=True).dt.tz_convert(None)

    return ohlc_df, sentiment_df, technical_df, llm_predictions_df, news_df

# Load data only once using cache
ohlc_df, sentiment_df, technical_df, llm_predictions_df, news_df = load_data()


# Sidebar - Stock Selection
st.sidebar.title("📊 Stock Market Dashboard")

# Stock Selector
selected_ticker = st.sidebar.selectbox("Choose a Stock:", ohlc_df["Ticker"].unique())

# Date Range Selector
start_date = st.sidebar.date_input("Start Date", value=ohlc_df["DateTime"].min())
end_date = st.sidebar.date_input("End Date", value=ohlc_df["DateTime"].max())

# Show selected filters
st.sidebar.write(f"📆 **Selected Date Range:** {start_date} to {end_date}")

# Filter Data
ohlc_filtered = ohlc_df[(ohlc_df["Ticker"] == selected_ticker) & (ohlc_df["DateTime"].between(start_date, end_date))]
sentiment_filtered = sentiment_df[(sentiment_df["Ticker"] == selected_ticker) & (sentiment_df["Date"].between(start_date, end_date))]
technical_filtered = technical_df[(technical_df["Ticker"] == selected_ticker) & (technical_df["DateTime"].between(start_date, end_date))]
llm_filtered = llm_predictions_df[(llm_predictions_df["Ticker"] == selected_ticker) & (llm_predictions_df["DateTime"].between(start_date, end_date))]
news_filtered = news_df[(news_df["Ticker"] == selected_ticker) & (news_df["DateTime"].between(start_date, end_date))]


# Optimize Stock Price Trend Chart
st.subheader(f"📈 Stock Price Trend for {selected_ticker}")
fig_price_trend = px.line(
    ohlc_filtered,
    x="DateTime",
    y="Close",
    title=f"📈 {selected_ticker} - Closing Price Trend",
    labels={"Close": "Closing Price ($)", "DateTime": "Date"},
    markers=True,
    template="plotly_dark"
)
st.plotly_chart(fig_price_trend, use_container_width=True)


# 📈 Live Stock Price Section
def get_live_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.history(period="1d")["Close"].values[-1]
    except:
        return "N/A"

# Display Live Stock Price
st.sidebar.subheader("📈 Live Stock Price")
live_price = get_live_price(selected_ticker)
st.sidebar.write(f"💰 **{selected_ticker} Current Price:** ${live_price:.2f}")



# Add Stunning Styling for Better UI
st.markdown("""
    <style>
        /* Global Settings */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f5f5f5;
        }

        /* Make Titles Stand Out */
        h1 {
            font-size: 34px !important;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
        }

        h2 {
            font-size: 28px !important;
            font-weight: bold;
            color: #333;
            border-bottom: 2px solid #4A90E2;
            padding-bottom: 5px;
        }

        h3 {
            font-size: 24px !important;
            font-weight: bold;
            color: #333;
        }

        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background-color: #1E293B !important;
            color: white !important;
        }

        [data-testid="stSidebar"] h2 {
            color: #4A90E2 !important;
        }

        [data-testid="stSidebarNav"] {
            background-color: #334155 !important;
        }

        /* Buttons Styling */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #4A90E2;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 8px;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #1E293B !important;
            transform: scale(1.05);
        }

        /* DataFrame Styling */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Centering Elements */
        .big-font {
            font-size: 20px !important;
            font-weight: bold;
            color: #333;
            text-align: center;
        }

        /* Container for Charts */
        .chart-container {
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        /* Metric Styling */
        .metric-box {
            padding: 10px;
            border-radius: 8px;
            background: linear-gradient(to right, #4A90E2, #1E293B);
            color: white;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }

        /* Hover Effects for Cards */
        .stMarkdown:hover, .stDataFrame:hover {
            transform: scale(1.02);
            transition: all 0.3s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)


