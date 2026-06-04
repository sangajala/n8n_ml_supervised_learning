import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("uk_weather_data.csv")
df["month"] = pd.to_datetime(df["date"]).dt.month

city_encoder = LabelEncoder()
df["city_enc"] = city_encoder.fit_transform(df["city"])

X = df[["city_enc", "month"]].values
y = df["temperature_celsius"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = DecisionTreeRegressor(max_depth=6, random_state=42)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Temperature Regressor — MAE: {mae:.2f}°C  |  R²: {r2:.3f}")

with open("temp_model.pkl", "wb") as f:
    pickle.dump({
        "model": reg,
        "city_encoder": city_encoder,
        "cities": list(city_encoder.classes_),
    }, f)

print("Saved to: temp_model.pkl")
