import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load UK weather dataset
df = pd.read_csv("uk_weather_data.csv")

# Feature engineering
df["month"] = pd.to_datetime(df["date"]).dt.month

# Encode city
city_encoder = LabelEncoder()
df["city_enc"] = city_encoder.fit_transform(df["city"])

# Encode target
label_encoder = LabelEncoder()
df["condition_enc"] = label_encoder.fit_transform(df["weather_condition"])

features = ["city_enc", "month", "temperature_celsius", "humidity_percent", "precipitation_mm"]
target = "condition_enc"

X = df[features].values
y = df[target].values
class_names = label_encoder.classes_

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
clf = DecisionTreeClassifier(max_depth=7, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 55)
print("  Decision Tree — UK Weather Dataset")
print("=" * 55)
print(f"\nDataset rows     : {len(df)}")
print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Features         : {', '.join(features)}")
print(f"Classes          : {', '.join(class_names)}")
print(f"\nTest Accuracy    : {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Save model + encoders for API use
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": clf,
        "city_encoder": city_encoder,
        "label_encoder": label_encoder,
        "feature_names": features,
        "cities": list(city_encoder.classes_),
        "classes": list(class_names),
    }, f)
print("Model saved to: model.pkl")

# Visualise tree
fig, ax = plt.subplots(figsize=(28, 12))
plot_tree(
    clf,
    feature_names=features,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
)
ax.set_title(
    "Decision Tree Classifier — UK Weather Conditions\n"
    f"Features: city, month, temperature, humidity, precipitation  |  Test Accuracy: {accuracy*100:.1f}%",
    fontsize=14,
    fontweight="bold",
    pad=16,
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
print("Tree graph saved to: decision_tree.png")
