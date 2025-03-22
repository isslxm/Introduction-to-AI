import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Загрузка данных
# Для примера используем сгенерированные данные
np.random.seed(42)
num_samples = 1000
data = {
    'temperature': np.random.uniform(15, 35, num_samples),  # Температура в градусах Цельсия
    'humidity': np.random.uniform(30, 90, num_samples),     # Влажность в %
    'pressure': np.random.uniform(980, 1030, num_samples),  # Давление в гПа
    'wind_speed': np.random.uniform(0, 15, num_samples),    # Скорость ветра в м/с
    'precipitation': np.random.uniform(0, 20, num_samples)  # Осадки в мм
}

df = pd.DataFrame(data)

# Подготовка данных
X = df[['humidity', 'pressure', 'wind_speed', 'precipitation']]
y = df['temperature']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Предсказание температуры на тестовых данных
y_pred = model.predict(X_test_scaled)

# Оценка точности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Вывод информации о модели
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coefficient of Determination (R^2): {r2:.4f}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_:.2f}")

# Визуализация предсказанных и фактических температур
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs Predicted Temperature')
plt.grid(True)
plt.show()

# # График распределения ошибок
# errors = y_test - y_pred
# plt.figure(figsize=(10, 6))
# plt.hist(errors, bins=20, color='purple', alpha=0.7, edgecolor='black')
# plt.xlabel('Prediction Error (°C)')
# plt.ylabel('Frequency')
# plt.title('Distribution of Prediction Errors')
# plt.grid(True)
# plt.show()
