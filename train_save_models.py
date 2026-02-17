import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score

# Load dataset
df = pd.read_csv('SriLanka_Weather_Dataset.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(['city', 'time'])

# 1. Preprocessing & Feature Engineering
df['month'] = df['time'].dt.month
df['previous_temp'] = df.groupby('city')['temperature_2m_max'].shift(0)
df['previous_rain'] = df.groupby('city')['rain_sum'].shift(0)
df['previous_wind'] = df.groupby('city')['windspeed_10m_max'].shift(0)

# Targets (Tomorrow's Forecast)
df['target_temp'] = df.groupby('city')['temperature_2m_max'].shift(-1)
df['target_rain'] = (df.groupby('city')['rain_sum'].shift(-1) > 0.5).astype(int)

data = df.dropna()

# 2. Encoding & Elevation mapping
city_elevations = data.groupby('city')['elevation'].first().to_dict()
city_dummies = pd.get_dummies(data['city'], prefix='city')
X = pd.concat([data[['previous_temp', 'previous_rain', 'previous_wind', 'month', 'elevation']], city_dummies], axis=1)

# 3. Algorithm Selection: Gradient Boosting
# Train/Test Split (Sequential to prevent look-ahead bias)
X_train, X_test, y_t_train, y_t_test = train_test_split(X, data['target_temp'], test_size=0.2, shuffle=False)
_, _, y_r_train, y_r_test = train_test_split(X, data['target_rain'], test_size=0.2, shuffle=False)

# Train Regression (Temp) and Classification (Rain)
temp_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5).fit(X_train, y_t_train)
rain_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5).fit(X_train, y_r_train)

# 4. Evaluation
print(f"Temp MAE: {mean_absolute_error(y_t_test, temp_model.predict(X_test)):.2f}")
print(f"Rain F1: {f1_score(y_r_test, rain_model.predict(X_test)):.2f}")


explainer = shap.TreeExplainer(temp_model)

with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)


shap.summary_plot(explainer.shap_values(X_test), X_test, show=False)

plt.savefig('shap_summary.png')
# 5. Save for Deployment 
pickle.dump(temp_model, open('temp_model.pkl', 'wb'))
pickle.dump(rain_model, open('rain_model.pkl', 'wb'))
pickle.dump(list(X.columns), open('model_columns.pkl', 'wb'))
pickle.dump(city_elevations, open('city_data.pkl', 'wb'))