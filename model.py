import pandas as pd
from prophet import Prophet
import pickle

# Load the dataset
df = pd.read_csv('apple_stock.csv')

# Data Cleaning and Preparation for Prophet
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
df.sort_values('ds', inplace=True)
df.dropna(inplace=True)

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df)

# Save the trained model to a file
model_filename = 'prophet_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Prophet model trained and saved as {model_filename}")