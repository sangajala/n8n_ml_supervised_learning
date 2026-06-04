import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

# Load sample dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split into train / test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("  Decision Tree — Iris Dataset")
print("=" * 50)
print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"\nTest Accuracy    : {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualise the tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=11,
    ax=ax,
)
ax.set_title("Decision Tree Classifier — Iris Dataset", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
print("Tree graph saved to: decision_tree.png")
plt.show()
