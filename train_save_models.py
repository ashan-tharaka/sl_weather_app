import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Load the full dataset
df = pd.read_csv('SriLanka_Weather_Dataset.csv')
df['time'] = pd.to_datetime(df['time'])

# 1. Feature Engineering per city
df = df.sort_values(['city', 'time'])
df['month'] = df['time'].dt.month
df['prev_temp'] = df.groupby('city')['temperature_2m_max'].shift(0)
df['prev_rain'] = df.groupby('city')['rain_sum'].shift(0)
df['prev_wind'] = df.groupby('city')['windspeed_10m_max'].shift(0)

# Targets
df['target_temp'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df['target_rain'] = (df.groupby('city')['rain_sum'].shift(-1) > 0.5).astype(int)
df['target_wind'] = df.groupby('city')['windspeed_10m_max'].shift(-1)

data = df.dropna()

# 2. One-Hot Encoding for Cities
city_dummies = pd.get_dummies(data['city'], prefix='city')
X = pd.concat([data[['prev_temp', 'prev_rain', 'prev_wind', 'month', 'elevation']], city_dummies], axis=1)

# Save column names to keep the order consistent in Flask
model_columns = list(X.columns)
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

# 3. Train & Save Models
temp_model = RandomForestRegressor(n_estimators=50).fit(X, data['target_temp'])
rain_model = RandomForestClassifier(n_estimators=50).fit(X, data['target_rain'])
wind_model = RandomForestRegressor(n_estimators=50).fit(X, data['target_wind'])

pickle.dump(temp_model, open('temp_model.pkl', 'wb'))
pickle.dump(rain_model, open('rain_model.pkl', 'wb'))
pickle.dump(wind_model, open('wind_model.pkl', 'wb'))

print("Multi-city models and column metadata saved!")