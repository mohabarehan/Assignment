import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
file_path = os.path.join(os.path.dirname(__file__), "DATA CSV.csv")
df = pd.read_csv(file_path, sep=";")
df = df.dropna(axis=1, how="all")
df = df.replace(r"^\s*(\d+)\D.*$", r"\1", regex=True)
df = df.replace(r"^\s*(\d+)\s*$", r"\1", regex=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(how="any").astype(int)
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))
example_features = X_test[0].reshape(1, -1)
example_pred = knn.predict(example_features)
print("Preference:", "iPhone" if example_pred[0] == 1 else "Android")