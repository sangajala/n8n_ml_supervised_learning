import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load UK weather dataset
df = pd.read_csv("uk_weather_data.csv")

# Features and target
features = ["temperature_celsius", "humidity_percent", "precipitation_mm"]
target = "weather_condition"

X = df[features].values
y_raw = df[target].values

# Encode string labels to integers
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

# Split into train / test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
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
print(f"Classes          : {', '.join(class_names)}")
print(f"\nTest Accuracy    : {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualise the tree
fig, ax = plt.subplots(figsize=(26, 12))
plot_tree(
    clf,
    feature_names=features,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
)
ax.set_title(
    "Decision Tree Classifier — UK Weather Conditions\n"
    f"Features: temperature, humidity, precipitation  |  Test Accuracy: {accuracy*100:.1f}%",
    fontsize=14,
    fontweight="bold",
    pad=16,
)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
print("Tree graph saved to: decision_tree.png")
plt.show()
