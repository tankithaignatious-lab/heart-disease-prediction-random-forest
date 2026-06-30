import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
df = pd.read_csv("heart-disease ITP (1) (1).csv")
df.head()
df.info()
df.isna().sum()
df["target"].value_counts()
X = df.drop(columns=["target"])
y = df["target"]
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred = rf_default.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest (Default) Confusion Matrix")
plt.show()
importances = rf_default.feature_importances_
imp_df = pd.DataFrame({"feature": X.columns, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="importance", y="feature", data=imp_df.head(10))
plt.title("Top 10 Feature Importances (Random Forest Default)")
plt.show()

imp_df.head(10)
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5]
}
grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",   # good for imbalanced/clinical data
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
best_rf = grid.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

print("Tuned Classification Report:\n")
print(classification_report(y_test, y_pred_tuned))

cm2 = confusion_matrix(y_test, y_pred_tuned)

plt.figure(figsize=(5,4))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest (Tuned) Confusion Matrix")
plt.show()
from sklearn.metrics import f1_score, accuracy_score

print("Default Accuracy:", accuracy_score(y_test, y_pred))
print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))

print("Default F1:", f1_score(y_test, y_pred))
print("Tuned F1:", f1_score(y_test, y_pred_tuned))