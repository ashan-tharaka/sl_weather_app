from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load objects
temp_model = pickle.load(open('temp_model.pkl', 'rb'))
rain_model = pickle.load(open('rain_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))
city_elevations = pickle.load(open('city_data.pkl', 'rb'))

@app.route('/')
def home():
    cities = sorted(city_elevations.keys())
    return render_template('index.html', cities=cities)

@app.route('/predict', methods=['POST'])
def predict():
    selected_city = request.form['city']
    
    # Construct input vector matching training data
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    input_df['previous_temp'] = float(request.form['temp'])
    input_df['previous_rain'] = float(request.form['rain'])
    input_df['previous_wind'] = float(request.form['wind'])
    input_df['month'] = int(request.form['month'])
    input_df['elevation'] = city_elevations.get(selected_city, 15.0)
    
    if f'city_{selected_city}' in input_df.columns:
        input_df[f'city_{selected_city}'] = 1

    # Prediction
    p_temp = temp_model.predict(input_df)[0]
    p_rain = rain_model.predict(input_df)[0]
    
    # Feature Importance for XAI 
    importance = temp_model.feature_importances_
    top_feature = model_columns[np.argmax(importance)]

    return render_template('index.html', 
                           cities=sorted(city_elevations.keys()),
                           res_city=selected_city,
                           temp=round(p_temp, 2), 
                           rain="Rain 🌧️" if p_rain == 1 else "Clear ☀️",
                           xai_note=f"Prediction influenced mostly by: {top_feature}")

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))