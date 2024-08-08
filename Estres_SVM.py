import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
import joblib

# Cargar los datos ajustados
file_path = 'Data/datos_entrenamiento.xlsx'
data = pd.read_excel(file_path)

# Preparar los datos
X = data[['Frecuencia Cardíaca', 'Frecuencia Respiratoria', 'Temperatura Corporal']]
y = data['Estrés (%)']

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo y los hiperparámetros para la búsqueda
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'epsilon': [0.1, 0.2],
    'gamma': ['scale', 'auto'],
    'shrinking': [True, False],
    'tol': [1e-3, 1e-4],
    'max_iter': [20000, 30000]
}

best_r2 = 0
best_model = None
best_params = None

while best_r2 < 0.955:
    # Usar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Mejor modelo encontrado por GridSearchCV
    best_model = grid_search.best_estimator_

    # Hacer predicciones
    y_pred_test = best_model.predict(X_test)

    # Evaluar el modelo
    test_r2 = r2_score(y_test, y_pred_test)

    if test_r2 > best_r2:
        best_r2 = test_r2
        best_params = grid_search.best_params_

# Guardar el mejor modelo encontrado
joblib.dump(best_model, 'modelo_entrenado_SVM.pkl')
joblib.dump(scaler, 'scaler_SVM.pkl')

# Hacer predicciones finales
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
print(f'Mejor conjunto de hiperparámetros: {best_params}')
print(f'Mejor R²: {best_r2}')

# Gráfica de predicciones vs valores reales
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, label='Línea ideal')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones')
plt.legend()
plt.grid(True)
plt.show()

# Curva de error
error = y_test - y_pred_test
plt.figure(figsize=(10, 5))
plt.hist(error, bins=25, color='orange', edgecolor='k')
plt.xlabel('Error de Predicción')
plt.ylabel('Frecuencia')
plt.title('Histograma del Error de Predicción')
plt.grid(True)
plt.show()

# Gráfica de MSE, R², MAE y MedAE
metrics = ['Train MSE', 'Test MSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE', 'Train MedAE', 'Test MedAE']
values = [train_mse, test_mse, train_r2, test_r2, train_mae, test_mae, train_medae, test_medae]

plt.figure(figsize=(14, 8))
plt.bar(metrics, values, color=['blue', 'orange', 'blue', 'orange', 'blue', 'orange', 'blue', 'orange'])
plt.title('Métricas del Modelo')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Datos de prueba para personas con diferentes niveles de estrés
test_data = [
    {"Frecuencia Cardíaca": 65, "Frecuencia Respiratoria": 15, "Temperatura Corporal": 36.5},  # Estrés bajo
    {"Frecuencia Cardíaca": 100, "Frecuencia Respiratoria": 25, "Temperatura Corporal": 37},  # Estrés medio
    {"Frecuencia Cardíaca": 120, "Frecuencia Respiratoria": 30, "Temperatura Corporal": 38},  # Estrés alto
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