import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("Генерация датасета...")
np.random.seed(42)
n_samples = 500

# Признаки:
# Age: Возраст авто (лет)
# Mileage: Пробег (тыс. км)
# Engine_Power: Мощность двигателя (л.с.)
# Brand_Index: Индекс престижности бренда (1-10)
# Accident_History: Количество ДТП (0-3)

age = np.random.randint(1, 20, n_samples)
mileage = np.random.randint(5, 300, n_samples)
engine_power = np.random.randint(80, 350, n_samples)
brand_index = np.random.randint(1, 11, n_samples)
accident_history = np.random.choice([0, 0, 0, 1, 1, 2], n_samples)

# Формула: Базовая цена + влияние мощности и бренда - износ от возраста и пробега - ущерб от ДТП
price = (
        10
        + (engine_power * 0.15)
        + (brand_index * 1.2)
        - (age * 0.8)
        - (mileage * 0.05)
        - (accident_history * 2.5)
        + np.random.normal(0, 2, n_samples)  # Добавляем шум
)

data = pd.DataFrame({
    'Age': age,
    'Mileage': mileage,
    'Engine_Power': engine_power,
    'Brand_Index': brand_index,
    'Accident_History': accident_history,
    'Price': price
})

print(f"Размер датасета: {data.shape}")
print(data.head())
print("\nОписательная статистика:")
print(data.describe())

X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

print("\nОбучение моделей и оценка на тестовой выборке...")
print("-" * 60)

for name, model in models.items():
    if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train_scaled, X_test_scaled

    # Обучение
    model.fit(X_tr, y_train)

    # Прогноз
    y_pred = model.predict(X_te)

    # Метрики
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_R2_Mean': cv_scores.mean(),
        'Predictions': y_pred
    }

    print(f"{name}:")
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    print(f"  Mean CV R2: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")


# 1. Сравнение метрик
model_names = list(results.keys())
r2_scores = [results[m]['R2'] for m in model_names]
rmse_scores = [results[m]['RMSE'] for m in model_names]

plt.figure(figsize=(12, 5))

# График R2 Score
plt.subplot(1, 2, 1)
bars = plt.bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title('R2 Score Comparison')
plt.ylabel('R2 Score')
plt.ylim(min(r2_scores) - 0.1, 1.0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center')

# График RMSE
plt.subplot(1, 2, 2)
bars = plt.bar(model_names, rmse_scores, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
plt.title('RMSE Comparison (Lower is better)')
plt.ylabel('RMSE')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{yval:.2f}', ha='center')

plt.tight_layout()
plt.show()

# 2. График "Факт vs Прогноз" для лучшей модели
best_model_name = max(results, key=lambda k: results[k]['R2'])
print(f"\nЛучшая модель по R2: {best_model_name}")

y_pred_best = results[best_model_name]['Predictions']

plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.title(f'Actual vs Predicted Price ({best_model_name})')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# 3. Важность признаков (для Random Forest, если он есть)
if "Random Forest" in models:
    rf_model = models["Random Forest"]
    # Получаем feature names
    feature_names = X.columns

    # Важно: если мы обучали RF на масштабированных данных, коэффициенты важности все равно корректны относительно структуры дерева
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train)
y_pred_pca = lr_pca.predict(X_test_pca)

r2_pca = r2_score(y_test, y_pred_pca)
print(f"\nPCA Analysis:")
print(f"Explained Variance Ratio by 2 components: {sum(pca.explained_variance_ratio_):.2f}")
print(f"R2 Score with PCA (Linear Regression): {r2_pca:.2f}")

# Визуализация данных в пространстве PCA
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.6)
plt.title('PCA Projection of Car Data (Colored by Price)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Price')
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()
