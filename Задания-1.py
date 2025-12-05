import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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

results = []
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((k, acc))
    accuracies.append(acc)
    print(f"k={k}, accuracy={acc:.4f}")

with open("results.txt", "w") as f:
    for k, acc in results:
        f.write(f"k={k}, accuracy={acc:.4f}\n")

plt.plot(k_values, accuracies, marker="o")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.grid(True)
plt.savefig("accuracy_vs_k.png")
plt.show()
