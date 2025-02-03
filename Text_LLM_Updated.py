import json
import seaborn as sns
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import openai
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, mean_squared_error
import google.generativeai as genai

tickers = ["META", "AAPL", "MSFT", "AMZN", "TSLA", "NVDA"]
start_date = "2024-11-01"
end_date = "2025-01-30"

def get_ohlc_data(tickers, start_date, end_date):
    ohlc_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval="1d")
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "Timestamp"}, inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Ticker"] = ticker
        ohlc_data[ticker] = df
    return ohlc_data

# Fetch OHLC Data
ohlc_data = get_ohlc_data(tickers, start_date, end_date)
ohlc_df = pd.concat(ohlc_data.values())

# Save the OHLC data
ohlc_df.to_csv("ohlc_data_cleaned.csv", index=False)


# Function to fetch news from Finnhub
def get_finnhub_news(ticker, start_date, end_date):
    FINNHUB_API_KEY = "cudomf9r01qiosq0olk0cudomf9r01qiosq0olkg"
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
    response = requests.get(url).json()

    if isinstance(response, list) and response:
        return [
            {
                "Ticker": ticker,
                "Title": article.get("headline", "N/A"),
                "Published Date": article.get("datetime"),
                "Summary": article.get("summary", "N/A"),
                "URL": article.get("url", "N/A")
            }
            for article in response
        ]
    return []


news_data = {ticker: get_finnhub_news(ticker, start_date, end_date) for ticker in tickers}
news_df = pd.concat([pd.DataFrame(v) for v in news_data.values()], ignore_index=True)
news_df["Published Date"] = pd.to_datetime(news_df["Published Date"], unit="s", errors="coerce")
news_df["Date"] = news_df["Published Date"].dt.date
news_df.to_csv("news_articles_cleaned.csv", index=False)

# Convert and clean timestamps
if "Published Date" in news_df.columns:
    news_df["Published Date"] = pd.to_datetime(news_df["Published Date"], unit="s", errors="coerce")
    news_df.dropna(subset=["Published Date"], inplace=True)

# Save Cleaned News Data
news_df.to_csv("news_articles_cleaned.csv", index=False)


MODEL_OPTIONS = {"FinBERT": "ProsusAI/finbert"}
sentiment_pipeline = pipeline("text-classification", model=MODEL_OPTIONS["FinBERT"])

def analyze_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral", 0.0

    result = sentiment_pipeline(text[:512])
    if isinstance(result, list) and len(result) > 0:
        sentiment_label = result[0].get("label", "neutral").lower()
        sentiment_score = result[0].get("score", 0.0)

        score_mapping = {"positive": 1, "neutral": 0, "negative": -1}
        sentiment_numeric = score_mapping.get(sentiment_label, 0)

        return sentiment_label, sentiment_numeric * sentiment_score
    return "neutral", 0.0

news_df["Sentiment"], news_df["Sentiment Score"] = zip(*news_df["Summary"].apply(analyze_sentiment))

sentiment_aggregated = news_df.groupby(["Ticker", "Date"]).agg(
    Sentiment_Score_Avg=("Sentiment Score", "mean"),
    Positive_News_Count=("Sentiment", lambda x: (x == "positive").sum()),
    Negative_News_Count=("Sentiment", lambda x: (x == "negative").sum()),
).reset_index()

sentiment_aggregated.to_csv("sentiment_aggregated.csv", index=False)

print("📊 Sentiment Aggregation Updated: Now includes sentiment scores per date!")


# Load Datasets
ohlc_df = pd.read_csv("ohlc_data_cleaned.csv")
sentiment_df = pd.read_csv("sentiment_aggregated.csv")

# Merge OHLC with Sentiment
merged_df = ohlc_df.merge(sentiment_df, on="Ticker", how="left")

# Save Final Merged Dataset
merged_df.to_csv("final_cleaned_dataset.csv", index=False)

def calculate_moving_averages(data, column="Close", short_window=10, long_window=15):
    """Calculates SMA & EMA"""
    data["SMA_10"] = data[column].rolling(window=short_window, min_periods=1).mean()
    data["SMA_15"] = data[column].rolling(window=long_window, min_periods=1).mean()
    data["EMA_10"] = data[column].ewm(span=short_window, adjust=False).mean()
    data["EMA_15"] = data[column].ewm(span=long_window, adjust=False).mean()
    return data

def calculate_bollinger_bands(data, column="Close", period=20):
    """
    Calculate Bollinger Bands:
    - Upper Band = SMA + (2 * Standard Deviation)
    - Lower Band = SMA - (2 * Standard Deviation)

    Parameters:
    - data: DataFrame containing stock price data
    - column: Column to compute Bollinger Bands on (default: 'Close')
    - period: Number of days to calculate moving average and standard deviation

    Returns:
    - DataFrame with Bollinger Upper and Lower Bands
    """
    data["SMA"] = data[column].rolling(window=period, min_periods=1).mean()
    data["StdDev"] = data[column].rolling(window=period, min_periods=1).std()

    # Calculate Upper and Lower Bollinger Bands
    data["Bollinger_Upper"] = data["SMA"] + (2 * data["StdDev"])
    data["Bollinger_Lower"] = data["SMA"] - (2 * data["StdDev"])

    # Drop intermediate columns to keep the DataFrame clean
    data.drop(columns=["SMA", "StdDev"], inplace=True)

    return data

def calculate_momentum(data, column="Close", period=10):
    """
    Calculate Momentum Indicator:
    - Momentum = Current Price - Price N Periods Ago

    Parameters:
    - data: DataFrame containing stock price data
    - column: Column to compute momentum on (default: 'Close')
    - period: Number of days to measure momentum

    Returns:
    - DataFrame with Momentum Indicator
    """
    data["Momentum"] = data[column] - data[column].shift(period)
    return data


def calculate_volatility(data, column="Close", period=20):
    """
    Calculate Volatility:
    - Volatility = Rolling Standard Deviation of Price Changes

    Parameters:
    - data: DataFrame containing stock price data
    - column: Column to compute volatility on (default: 'Close')
    - period: Number of days for rolling standard deviation

    Returns:
    - DataFrame with Volatility Indicator
    """
    data["Volatility"] = data[column].pct_change().rolling(window=period, min_periods=1).std()
    return data


# Apply Moving Averages to merged_df
merged_df = calculate_moving_averages(merged_df)

# Confirm the column now exists
if "SMA_10" in merged_df.columns:
    print("✅ 'SMA_10' successfully added.")
else:
    print("❌ 'SMA_10' is still missing!")

# Function to calculate RSI
def calculate_rsi(data, column="Close", period=14):
    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD
def calculate_macd(data, column="Close"):
    short_ema = data[column].ewm(span=12, adjust=False).mean()
    long_ema = data[column].ewm(span=26, adjust=False).mean()
    data["MACD"] = short_ema - long_ema
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data

# Apply Technical Indicators
merged_df = calculate_rsi(merged_df)
merged_df = calculate_macd(merged_df)
# Apply Technical Indicators
merged_df = calculate_bollinger_bands(merged_df)
merged_df = calculate_momentum(merged_df)
merged_df = calculate_volatility(merged_df)

# Save Updated Dataset
merged_df.to_csv("technical_indicators_dataset.csv", index=False)


# Set up Google Gemini API Key
genai.configure(api_key="AIzaSyDsqfG6sKuJnNnc71Sj_vhBuDjGfFj1Sds")

# Function to generate structured LLM prompts
def create_prompt(row):
    return f"""
    Stock Analysis Request:

    Stock: {row['Ticker']}
    Date: {row['Timestamp']}
    Closing Price: ${row['Close']:.2f}

    Technical Indicators:
    - RSI: {row['RSI']:.2f}
    - SMA (10-day): {row['SMA_10']:.2f}
    - SMA (15-day): {row['SMA_15']:.2f}
    - EMA (10-day): {row['EMA_10']:.2f}
    - EMA (15-day): {row['EMA_15']:.2f}
    - Bollinger Bands: Upper {row['Bollinger_Upper']:.2f}, Lower {row['Bollinger_Lower']:.2f}
    - MACD: {row['MACD']:.2f} | Signal Line: {row['MACD_Signal']:.2f}
    - Volatility: {row['Volatility']:.2f}
    - Momentum: {row['Momentum']:.2f}

    Market Sentiment:
    - Sentiment Score: {row['Sentiment_Score_Avg']:.2f}
    - Positive News Articles: {row['Positive_News_Count']}
    - Negative News Articles: {row['Negative_News_Count']}

    Based on the above data, analyze the stock’s short-term movement.
    Will the stock price increase, decrease, or remain stable over the next few days?
    Provide reasoning based on the indicators.
    """

# Apply prompt function
merged_df["LLM_Prompt"] = merged_df.apply(create_prompt, axis=1)

# Save dataset with prompts
merged_df.to_csv("llm_prompts_dataset.csv", index=False)
print("📜 LLM Prompts Saved!")


# Function to get LLM response
def get_llm_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error generating LLM response: {e}")
        return "Error in fetching response"

# Apply OpenAI API to all prompts
merged_df["LLM_Response"] = merged_df["LLM_Prompt"].apply(get_llm_response)

# Save dataset with responses
merged_df.to_csv("llm_responses_dataset.csv", index=False)

print("📄 All LLM Responses Retrieved and Saved!")


def quantify_response(response, stock_ticker):
    """
    Converts LLM response into numerical predictions (+1 = Bullish, -1 = Bearish, 0 = Neutral)
    with confidence adjustment using stock volatility.
    """

    # Define sentiment indicators and confidence keywords
    bullish_keywords = ["increase", "rise", "bullish", "positive trend", "strong growth"]
    bearish_keywords = ["decrease", "fall", "bearish", "negative trend", "decline"]
    neutral_keywords = ["stable", "sideways movement", "uncertain", "no major change"]

    confidence_high = ["high confidence", "very likely", "strong probability"]
    confidence_medium = ["moderate confidence", "reasonable probability", "likely"]
    confidence_low = ["low confidence", "some chance", "uncertain"]

    # Convert response to lowercase
    response_lower = response.lower()

    # Initialize sentiment and confidence scores
    sentiment_score = 0
    confidence_score = 0.5  # Default confidence

    # Detect sentiment direction
    bullish_count = sum(1 for word in bullish_keywords if word in response_lower)
    bearish_count = sum(1 for word in bearish_keywords if word in response_lower)
    neutral_count = sum(1 for word in neutral_keywords if word in response_lower)

    if bullish_count > bearish_count and bullish_count > neutral_count:
        sentiment_score = 1  # Bullish
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        sentiment_score = -1  # Bearish
    else:
        sentiment_score = 0  # Neutral

    # Assign confidence score
    if any(word in response_lower for word in confidence_high):
        confidence_score = 0.9
    elif any(word in response_lower for word in confidence_medium):
        confidence_score = 0.7
    elif any(word in response_lower for word in confidence_low):
        confidence_score = 0.3

    # Fetch stock volatility from Yahoo Finance
    try:
        df_vol = yf.download(stock_ticker, period="6mo", interval="1d")
        df_vol["Volatility"] = df_vol["Close"].rolling(window=20).std()
        latest_volatility = df_vol["Volatility"].dropna().iloc[-1]

        # Adjust confidence based on volatility
        confidence_score *= (1 + latest_volatility)
        confidence_score = max(0, min(confidence_score, 1))

    except Exception:
        print(f"⚠️ Warning: Could not fetch volatility for {stock_ticker}. Using default confidence.")

    return sentiment_score, confidence_score

# Apply quantification
merged_df["LLM_Prediction"], merged_df["LLM_Confidence"] = zip(
    *merged_df.apply(lambda row: quantify_response(row["LLM_Response"], row["Ticker"]), axis=1)
)
merged_df.drop(columns=["LLM_Prompt", "LLM_Response"], inplace=True)
# Save final dataset with quantified predictions
merged_df.to_csv("llm_quantified_predictions.csv", index=False)

print("📊 Final Dataset with Quantified Predictions Saved!")


from sklearn.metrics import accuracy_score, classification_report

# Load final dataset
df = pd.read_csv("llm_quantified_predictions.csv")

df["Actual_Price_Change"] = df.groupby("Ticker")["Close"].pct_change() * 100
df["Actual_Trend"] = df["Actual_Price_Change"].apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))

accuracy = accuracy_score(df["Actual_Trend"], df["LLM_Prediction"]) * 100
precision = precision_score(df["Actual_Trend"], df["LLM_Prediction"], average="weighted")
recall = recall_score(df["Actual_Trend"], df["LLM_Prediction"], average="weighted")
f1 = (2 * precision * recall) / (precision + recall)
rmse = np.sqrt(mean_squared_error(df["Actual_Trend"], df["LLM_Prediction"]))

# ✅ Display Results
print(f"✅ Accuracy: {accuracy:.2f}%")
print(f"📍 Precision: {precision:.2f}")
print(f"📍 Recall: {recall:.2f}")
print(f"📍 F1 Score: {f1:.2f}")
print(f"📍 RMSE: {rmse:.2f}")

# ✅ Sector-Wise Stock Performance
stock_performance = df.groupby("Ticker").agg(
    Accuracy=("Actual_Trend", "mean"),
    Avg_Confidence=("LLM_Confidence", "mean"),
).reset_index()
