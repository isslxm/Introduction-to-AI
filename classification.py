import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Генерация примерных данных о пациентах
np.random.seed(42)
num_samples = 200

# Признаки: уровень сахара в крови, давление
X = np.random.rand(num_samples, 2) * 100

y = ((X[:, 0] > 80) & (X[:, 1] > 85)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Предсказания и оценка модели
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Вывод отчета о классификации
print("Classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Вывод матрицы ошибок
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib
# Сохранение модели
joblib.dump(clf, 'diabetes_model.pkl')


# Загрузка модели
clf_loaded = joblib.load('diabetes_model.pkl')
print("Model downloaded.")

samples = np.array([
    [70, 80],  # скорее всего нет риска
    [85, 90],  # скорее всего есть риск
    [60, 60],  # нет риска
    [95, 88],  # риск
])

predictions = clf_loaded.predict(samples)
for i, pred in enumerate(predictions):
    print(f"Patient {i+1}: {'Risk' if pred else 'No risk'}")

