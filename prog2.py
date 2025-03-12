import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных (используем California Housing dataset)
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target * 100000  # Переводим в доллары для наглядности

# Подготовка данных
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание цен на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 100 * (1 - np.mean(np.abs((y_test - y_pred) / y_test)))

# Создаем один наглядный график
plt.figure(figsize=(10, 6))

# Получаем индексы для сортировки фактических цен
sorted_indices = np.argsort(y_test.values)
x_values = np.arange(len(sorted_indices))

# Рисуем фактические и предсказанные цены в порядке возрастания фактических цен
plt.scatter(x_values, y_test.values[sorted_indices], color='blue', alpha=0.7, label='Фактические цены')
plt.scatter(x_values, y_pred[sorted_indices], color='green', alpha=0.7, label='Предсказанные цены')

# Добавляем линию регрессии
z = np.polyfit(y_test.values[sorted_indices], y_pred[sorted_indices], 1)
p = np.poly1d(z)
plt.plot(x_values, p(y_test.values[sorted_indices]), color='green', linestyle='--', 
         label=f'Линия регрессии (y = {z[0]:.2f}x + {z[1]:.2f})')

# Добавляем оформление графика
plt.xlabel('Индекс наблюдения (отсортированный по фактической цене)')
plt.ylabel('Цена ($)')
plt.title(f'Сравнение фактических и предсказанных цен на жилье\nТочность: {accuracy:.2f}%, R^2: {r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Добавляем текст с информацией о точности
plt.figtext(0.5, 0.01, f'Среднеквадратическая ошибка (MSE): {mse:.2f}', 
            ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Вывод информации о модели
print(f"Коэффициент детерминации (R^2): {r2:.4f}")
print(f"Среднеквадратическая ошибка (MSE): {mse:.2f}")
print(f"Процентная точность: {accuracy:.2f}%")
print(f"Коэффициенты модели: {model.coef_}")
print(f"Свободный член: {model.intercept_:.2f}")