import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset/bank.csv", sep=";")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTarget Variable Distribution:")
print(df["y"].value_counts())
print("\nTarget Variable Percentage:")
print(df["y"].value_counts(normalize=True) * 100)
X = df.drop("y", axis=1)
y = df["y"]
X = pd.get_dummies(X, drop_first=True)
print("\nEncoded Dataset Shape:")
print(X.shape)
print("\nEncoded Feature Columns:")
print(X.columns.tolist())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Data Shape:")
print(X_train.shape)
print("\nTesting Data Shape:")
print(X_test.shape)
print("\nTraining Target Shape:")
print(y_train.shape)
print("\nTesting Target Shape:")
print(y_test.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nDecision Tree Accuracy:")
print(f"{accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.tree import plot_tree
import os
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Classifier - Bank Marketing")
plt.tight_layout()
plt.savefig(
    "outputs/decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nFeature Importance:")
print(importance)
plt.figure(figsize=(12, 7))
importance.head(15).sort_values().plot(kind="barh")
plt.title("Top 15 Feature Importances - Bank Marketing")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    "outputs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Bank Marketing")
plt.tight_layout()
plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()