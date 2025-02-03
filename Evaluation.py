# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, classification_report

# Load the dataset
file_path = "llm_quantified_predictions.csv"  # Change to your file path if needed
df = pd.read_csv(file_path)

# Calculate actual price change and trend
df["Actual_Price_Change"] = df.groupby("Ticker")["Close"].pct_change() * 100
df["Actual_Trend"] = df["Actual_Price_Change"].apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))

# Evaluate the model's performance
accuracy = accuracy_score(df["Actual_Trend"], df["LLM_Prediction"]) * 100
precision = precision_score(df["Actual_Trend"], df["LLM_Prediction"], average="weighted")
recall = recall_score(df["Actual_Trend"], df["LLM_Prediction"], average="weighted")
f1 = (2 * precision * recall) / (precision + recall)
rmse = np.sqrt(mean_squared_error(df["Actual_Trend"], df["LLM_Prediction"]))

# Display results
print(f"✅ Accuracy: {accuracy:.2f}%")
print(f"📍 Precision: {precision:.2f}")
print(f"📍 Recall: {recall:.2f}")
print(f"📍 F1 Score: {f1:.2f}")
print(f"📍 RMSE: {rmse:.2f}")

# Sector-Wise Stock Performance
stock_performance = df.groupby("Ticker").agg(
    Accuracy=("Actual_Trend", "mean"),
    Avg_Confidence=("LLM_Confidence", "mean"),
).reset_index()
