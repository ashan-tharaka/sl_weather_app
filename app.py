from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models and the exact column structure
temp_model = pickle.load(open('temp_model.pkl', 'rb'))
rain_model = pickle.load(open('rain_model.pkl', 'rb'))
wind_model = pickle.load(open('wind_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Elevation dictionary mapping for accuracy
city_info = {
    'Colombo': 16.0, 'Kandy': 500.0, 'Nuwara Eliya': 1889.0, 'Jaffna': 5.0,
    'Galle': 0.0, 'Ratnapura': 21.0, 'Badulla': 680.0 # Add more from your CSV
}

@app.route('/')
def home():
    # Get city names from the column list
    cities = [c.replace('city_', '') for c in model_columns if c.startswith('city_')]
    return render_template('index.html', cities=sorted(cities))

@app.route('/predict', methods=['POST'])
def predict():
    selected_city = request.form['city']
    
    # Initialize a dataframe with zeros matching the training columns
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Fill numerical values
    input_df['prev_temp'] = float(request.form['temp'])
    input_df['prev_rain'] = float(request.form['rain'])
    input_df['prev_wind'] = float(request.form['wind'])
    input_df['month'] = int(request.form['month'])
    input_df['elevation'] = city_info.get(selected_city, 15.0)
    
    # Set the specific city's binary column to 1
    city_col = f'city_{selected_city}'
    if city_col in input_df.columns:
        input_df[city_col] = 1

    # Predict
    p_temp = temp_model.predict(input_df)[0]
    p_rain = rain_model.predict(input_df)[0]
    p_wind = wind_model.predict(input_df)[0]

    cities = [c.replace('city_', '') for c in model_columns if c.startswith('city_')]
    return render_template('index.html', 
                           cities=sorted(cities),
                           res_city=selected_city,
                           temp=round(p_temp, 2), 
                           rain="Rain 🌧️" if p_rain == 1 else "Clear ☀️", 
                           wind=round(p_wind, 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)