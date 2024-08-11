# Importación de librerías
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 8
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Cargar los datos
file_path = 'Data/datos_entrenamiento.xlsx'  # Ajusta la ruta según la ubicación de tu archivo
data = pd.read_excel(file_path)

# Selección de variables: Edad, Frecuencia Cardíaca, Frecuencia Respiratoria, Temperatura Corporal, y Estrés (%)
X = data[['Frecuencia Cardíaca', 'Frecuencia Respiratoria', 'Temperatura Corporal', 'Edad']]
y = data['Estrés (%)']

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo y los hiperparámetros para la búsqueda
param_grid = {
    'n_estimators': [500],
    'criterion': ['squared_error'],
    'max_depth': [10],
    'min_samples_split': [10],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'bootstrap': [True],
}

# Búsqueda del mejor modelo y evaluación
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado por GridSearchCV
best_model = grid_search.best_estimator_

# Hacer predicciones
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Evaluar el modelo final
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_medae = median_absolute_error(y_train, y_pred_train)
test_medae = median_absolute_error(y_test, y_pred_test)

# Mostrar resultados de evaluación
print(f'Train MSE: {train_mse}, Train R²: {train_r2}')
print(f'Test MSE: {test_mse}, Test R²: {test_r2}')
print(f'Train MAE: {train_mae}, Test MAE: {test_mae}')
print(f'Train MedAE: {train_medae}, Test MedAE: {test_medae}')

# Gráfica de predicciones vs valores reales (Figura 1)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Línea ideal')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones ')
plt.legend()
plt.grid(True)
plt.show()

# Curva de error (Figura 2)
error = y_test - y_pred_test
plt.figure(figsize=(10, 5))
plt.hist(error, bins=25, color='orange', edgecolor='k')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Histograma del Error de Predicción ')
plt.grid(True)
plt.show()

# Optimización de hiperparámetros - Número de Árboles
train_scores = []
oob_scores   = []
estimator_range = range(1, 150, 5)

for n_estimators in estimator_range:
    modelo = RandomForestRegressor(
                n_estimators = n_estimators,
                criterion    = 'squared_error',
                max_depth    = None,
                max_features = 1,
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123
             )
    modelo.fit(X_train, y_train)
    train_scores.append(modelo.score(X_train, y_train))
    oob_scores.append(modelo.oob_score_)

# Gráfico de la evolución del out-of-bag-error vs número de árboles (Figura 3)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, oob_scores, label="out-of-bag scores")
ax.plot(estimator_range[np.argmax(oob_scores)], max(oob_scores),
        marker='o', color="red", label="max score")
ax.set_ylabel("R^2")
ax.set_xlabel("n_estimators")
ax.set_title("Evolución del out-of-bag-error vs número árboles ")
plt.legend()
plt.show()

# Validación empleando k-cross-validation y neg_root_mean_squared_error (Figura 4)
train_scores = []
cv_scores    = []
for n_estimators in estimator_range:
    modelo = RandomForestRegressor(
                n_estimators = n_estimators,
                criterion    = 'squared_error',
                max_depth    = None,
                max_features = 1,
                oob_score    = False,
                n_jobs       = -1,
                random_state = 123
             )
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X=X_train)
    rmse = mean_squared_error(y_true=y_train, y_pred=predicciones, squared=False)
    train_scores.append(rmse)
    scores = cross_val_score(
                estimator = modelo,
                X         = X_train,
                y         = y_train,
                scoring   = 'neg_root_mean_squared_error',
                cv        = 5
             )
    cv_scores.append(-1*scores.mean())

# Gráfico con la evolución del error cv vs número de árboles (Figura 4)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(estimator_range, train_scores, label="train scores")
ax.plot(estimator_range, cv_scores, label="cv scores")
ax.plot(estimator_range[np.argmin(cv_scores)], min(cv_scores),
        marker='o', color="red", label="min score")
ax.set_ylabel("root_mean_squared_error")
ax.set_xlabel("n_estimators")
ax.set_title("Evolución del cv-error vs número árboles ")
plt.legend()
plt.show()

# Importancia de los predictores - Permutación (Figura 5)
importancia = permutation_importance(
    estimator    = best_model,
    X            = X_train,
    y            = y_train,
    n_repeats    = 5,
    scoring      = 'neg_root_mean_squared_error',
    n_jobs       = cpu_count() - 1,
    random_state = 123
)
df_importancia = pd.DataFrame({
    'importances_mean': importancia['importances_mean'],
    'importances_std': importancia['importances_std'],
    'feature': X.columns
})
df_importancia = df_importancia.sort_values('importances_mean', ascending=True)

# Gráfico de la importancia de los predictores (Permutación) (Figura 5)
fig, ax = plt.subplots(figsize=(3.5, 4))
ax.barh(df_importancia['feature'], df_importancia['importances_mean'], xerr=df_importancia['importances_std'], align='center', alpha=0)
ax.plot(df_importancia['importances_mean'], df_importancia['feature'], marker="D", linestyle="", alpha=0.8, color="r")
ax.set_title('Importancia de los predictores (Permutación) ')
ax.set_xlabel('Incremento del error tras la permutación')
plt.show()

# Gráfica de MSE, R², MAE y MedAE (Figura 6)
metrics = ['Train MSE', 'Test MSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE', 'Train MedAE', 'Test MedAE']
values = [train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_medae, test_medae]

plt.figure(figsize=(14, 8))
plt.bar(metrics, values, color=['blue', 'orange', 'blue', 'orange', 'blue', 'orange', 'blue', 'orange'])
plt.title('Métricas del Modelo ')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Datos de prueba para personas con diferentes niveles de estrés incluyendo edad
test_data = [
    {"Frecuencia Cardíaca": 65, "Frecuencia Respiratoria": 15, "Temperatura Corporal": 36.5, "Edad": 22},  # Estrés bajo
    {"Frecuencia Cardíaca": 100, "Frecuencia Respiratoria": 25, "Temperatura Corporal": 37, "Edad": 30},  # Estrés medio
    {"Frecuencia Cardíaca": 120, "Frecuencia Respiratoria": 30, "Temperatura Corporal": 38, "Edad": 45},  # Estrés alto
]

# Convertir los datos de prueba a DataFrame
test_df = pd.DataFrame(test_data)

# Normalizar los datos de prueba
test_df_scaled = scaler.transform(test_df)

# Hacer predicciones con el modelo entrenado
stress_predictions = best_model.predict(test_df_scaled)

print("Predicciones de estrés para datos de prueba:")
for i, pred in enumerate(stress_predictions):
    print(f"Persona {i+1}: {pred:.2f}% de estrés")
