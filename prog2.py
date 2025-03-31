import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import fetch_california_housing

# Загрузка данных
california = fetch_california_housing()
print(california)
df = pd.DataFrame(california.data, columns=california.feature_names)
print(df.head)
df['PRICE'] = california.target * 100000  # Переводим в доллары для наглядности

# Подготовка данных
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Добавим полиномиальные признаки для учета нелинейных зависимостей
# Это повысит точность линейной регрессии, учитывая взаимодействия между признаками
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.05, random_state=42)

# Масштабирование признаков для стабильности модели
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели линейной регрессии на улучшенных данных
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Предсказание цен на тестовых данных
y_pred = model.predict(X_test_scaled)

# Оценка точности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 100 * (1 - np.mean(np.abs((y_test - y_pred) / y_test)))


# Создаем улучшенный наглядный график
plt.figure(figsize=(12, 7))

# Получаем индексы для сортировки фактических цен
sorted_indices = np.argsort(y_test.values)
x_values = np.arange(len(sorted_indices))

# Рисуем фактические и предсказанные цены в порядке возрастания фактических цен
plt.scatter(x_values, y_test.values[sorted_indices], color='blue', alpha=0.7, label='Actual prices')
# plt.scatter(x_values, y_pred[sorted_indices], color='green', alpha=0.7, label='Predicted prices')

# Добавляем линию регрессии
z = np.polyfit(y_test.values[sorted_indices], y_pred[sorted_indices], 1)
p = np.poly1d(z)
plt.plot(x_values, p(y_test.values[sorted_indices]), color='red', linestyle='--', 
         label=f'Linear regression (y = {z[0]:.2f}x + {z[1]:.2f})')

# Добавляем оформление графика
plt.xlabel('Observation index', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.title(f'Linear regression & actual prices\nAccuracy: {accuracy:.2f}%, R^2: {r2:.4f}', 
          fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Добавляем текст с информацией о точности
plt.figtext(0.5, 0.01, f'Mean squared error (MSE): {mse:.2f}', 
            ha='center', fontsize=11)

# Добавляем область доверительного интервала
plt.fill_between(x_values, 
                 y_pred[sorted_indices] - 0.5 * np.std(y_test - y_pred), 
                 y_pred[sorted_indices] + 0.5 * np.std(y_test - y_pred), 
                 color='green', alpha=0.1)

plt.tight_layout()
plt.show()

# Вывод информации о модели
print(f"Coefficient (R^2): {r2:.4f}")
print(f"Mean squared error (MSE): {mse:.2f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Model intercept: {model.intercept_:.2f}")
