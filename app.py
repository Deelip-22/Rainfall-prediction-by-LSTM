from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import pandas as pd
from datetime import datetime

#Load the trained model
model = keras.models.load_model('my_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Define the last date
    last_date = pd.Timestamp('2024-04-14')
    # Generate 30 days after the last date
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    # Convert future_dates to list
    future_dates_list = future_dates.tolist()

    # Load the data from normalized_data.npy
    normalized_data = np.load('X_test.npy')
    num_fut = 30
    forecast_pred_proba = model.predict(normalized_data[-num_fut:])
    forecast_pred = (forecast_pred_proba.flatten() > 0.5).astype(int)

    # Create a DataFrame with future_dates_list and forecast_pred
    df_forecast = pd.DataFrame({
        'Date': future_dates_list,
        'Rainfall_Predicted': forecast_pred
    })
    # Get the current date
    current_date = datetime.now().date()

    # Format the current date as a string
    current_date_str = current_date.strftime('%Y-%m-%d')


    # Filter the DataFrame for the next 7 days from the current date
    forecast_7_days = df_forecast[df_forecast['Date'] >= current_date_str][:7]
    # Add day of the week to the DataFrame
    forecast_7_days['Day'] = forecast_7_days['Date'].dt.day_name()
   

    return render_template('predict.html', forecast=forecast_7_days.to_dict(orient='records'))
if __name__ == '__main__':
    app.run(debug=True)
