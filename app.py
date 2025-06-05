from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__, template_folder='template')

# Load the trained Prophet model
model_filename = 'prophet_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found. Please run model.py first.")
    exit()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        date_str = request.form['date']
        try:
            future = pd.DataFrame([{'ds': pd.to_datetime(date_str)}])
        except ValueError:
            return render_template('index.html', error='Invalid date format. Please use "YYYY-MM-DD"')

        # Make prediction
        forecast = model.predict(future)

        # Extract the predicted value
        predicted_price = forecast['yhat'][0]

        return render_template('index.html', prediction=predicted_price, date=date_str)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)