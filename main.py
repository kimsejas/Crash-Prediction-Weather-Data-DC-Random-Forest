import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

def modelEvalResults(target: str, features: list, df: pd.DataFrame):
  y = df[target].values
  x = df[features].values
  scaler = StandardScaler()
  scaled_x = scaler.fit_transform(x)
  x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, random_state=24, test_size=0.2)
  rf = RandomForestClassifier(class_weight="balanced")
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  print(rf.score(x_test, y_test))
  confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

  # Visualize the confusion matrix using a heatmap
  plt.figure(figsize=(8, 6))
  sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
              xticklabels=['Predicted No Crash', 'Predicted Crash'],
              yticklabels=['Actual No Crash', 'Actual Crash'])
  plt.xlabel('Predicted Label')
  plt.ylabel('Actual Label')
  plt.title('Baseline Model Confusion Matrix')
  plt.show()

  print(classification_report(y_test, y_pred, target_names=['no crash', 'crash']))
  print("\n")

def main():
  # Load and preprocess data
  crashes = pd.read_csv("data/dc_crashes.csv", dtype={"CCN": str, "EVENTID": str}, parse_dates=["REPORTDATE"])
  weather = pd.read_csv("data/weather.csv", parse_dates=["DATE"])

  # Replace precipitation T (trace amounts of precip) values with 0.005 inches
  weather['HourlyPrecipitation'] = weather['HourlyPrecipitation'].replace('T', 0.005)

  # Filter for dates, temperature, precipitation, and wind speed
  columns = ['DATE', 'HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyWindSpeed']
  weather = weather.loc[:, columns]
  weather = weather.dropna()

  # Round down to the nearest hour to merge
  crashes.loc[:, "REPORTDATE"] = crashes["REPORTDATE"].dt.floor('h')
  weather.loc[:, "DATE"] = weather["DATE"].dt.floor('h')

  # Crash data is in UTC so convert
  crashes["REPORTDATE"] = pd.to_datetime(crashes["REPORTDATE"].dt.strftime('%Y-%m-%d %H:%M:%S'))

  # Left join on date and timestamp, capturing weather on times where no crashes have occurred
  crash_weather = pd.merge(weather, crashes, left_on='DATE', right_on='REPORTDATE', how='left')

  # Add a binary variable, 1 if crash occurred that time, 0 otherwise
  crash_weather['CrashOccured'] = crash_weather['CRIMEID'].notna().astype(int)

  # Add variables for hour and day of week
  crash_weather['Hour'] = crash_weather['DATE'].dt.hour  # Hour of the day (0-23)
  crash_weather['DayOfWeek'] = crash_weather['DATE'].dt.weekday  # Monday = 0, Sunday = 6

  # Create and print model results
  print("Model without Weather Data")
  modelEvalResults('CrashOccured', ['Hour', 'DayOfWeek'], crash_weather)

  print("Model with Weather Data")
  modelEvalResults('CrashOccured', ['HourlyDryBulbTemperature', 'HourlyPrecipitation', 'HourlyWindSpeed', 'Hour', 'DayOfWeek'], crash_weather)

  print("Model with Temperature Data Only")
  modelEvalResults('CrashOccured', ['HourlyDryBulbTemperature', 'Hour', 'DayOfWeek'], crash_weather)

  print("Model with Precipitation Data Only")
  modelEvalResults('CrashOccured', ['HourlyPrecipitation', 'Hour', 'DayOfWeek'], crash_weather)

  print("Model with Wind Data Only")
  modelEvalResults('CrashOccured', ['HourlyWindSpeed', 'Hour', 'DayOfWeek'], crash_weather)

if __name__ == "__main__":
  main()