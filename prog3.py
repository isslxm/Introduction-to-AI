import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Генерация данных о технике
np.random.seed(42)
num_samples = 1000

# Генерация исходных данных
data = {
    'condition': np.random.choice(['new', 'like_new', 'good', 'fair', 'poor'], num_samples),
    'brand': np.random.choice(['Apple', 'Samsung', 'Dell', 'Lenovo', 'Asus', 'HP'], num_samples),
    'age_years': np.random.uniform(0, 8, num_samples),
    'ram_gb': np.random.choice([4, 8, 16, 32, 64], num_samples),
    'storage_gb': np.random.choice([128, 256, 512, 1024, 2048], num_samples),
    'quantity': np.random.randint(1, 10, num_samples)
}

df = pd.DataFrame(data)

# Создание зависимости цены от параметров
# Базовая цена
base_price = 500

# Влияние состояния на цену
condition_effect = {
    'new': 1.0,
    'like_new': 0.85,
    'good': 0.7,
    'fair': 0.5,
    'poor': 0.3
}
df['condition_value'] = df['condition'].map(condition_effect)

# Влияние бренда на цену
brand_effect = {
    'Apple': 1.5,
    'Samsung': 1.3,
    'Dell': 1.1,
    'Lenovo': 1.0,
    'Asus': 0.9,
    'HP': 0.95
}
df['brand_value'] = df['brand'].map(brand_effect)

# Генерация цены с учетом всех факторов
df['price'] = (
    base_price +
    df['condition_value'] * 500 +
    df['brand_value'] * 300 +
    df['ram_gb'] * 10 +
    df['storage_gb'] * 0.5 +
    df['age_years'] * 100 +
    np.random.normal(0, 100, num_samples)  # Добавляем некоторый шум
) * df['quantity']  # Умножаем на количество

# Удаляем промежуточные столбцы
df = df.drop(['condition_value', 'brand_value'], axis=1)

print("Data:")
print(df.head())

# Разделение на признаки и целевую переменную
X = df.drop('price', axis=1)
y = df['price']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание препроцессора данных
categorical_features = ['condition', 'brand']
numeric_features = ['age_years', 'ram_gb', 'storage_gb', 'quantity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Создание пайплайна с препроцессором и моделью
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Обучение модели
model.fit(X_train, y_train)

# Предсказание цен на тестовых данных
y_pred = model.predict(X_test)

# Оценка точности модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = 100 * (1 - np.mean(np.abs((y_test - y_pred) / y_test)))

# Вывод информации о модели
print(f"Mean squared error (MSE): {mse:.2f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Observation index (R^2): {r2:.4f}")
# sample = np.array([['good', 'HP', 5, 16, 256, 8]])
# sample_pred = model.predict(sample)
# print(sample_pred)
# Визуализация предсказанных и фактических цен
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual price (som)')
plt.ylabel('Predicted price (som)')
plt.title(f"Actaul price vs predicted price\nAccuracy: {accuracy:.2f}%, R^2: {r2:.4f}",
          fontsize=14)
plt.grid(True)
plt.show()