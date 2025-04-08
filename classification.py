import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
num_samples = 200

# Признаки: уровень сахара в крови, давление
X = np.random.rand(num_samples, 2) * 100
y = ((X[:, 0] > 50) & (X[:, 1] > 70)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

print("Classification report:")
print(classification_report(y_test, y_pred))

# Вывод матрицы ошибок
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
