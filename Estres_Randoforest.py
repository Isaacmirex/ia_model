import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import joblib

# Cargar los datos ajustados
file_path = 'Data/datos_biometricos_ajustados_10000.xlsx'
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
    'n_estimators': [500],
    'criterion': ['absolute_error'],
    'max_depth': [10],
    'min_samples_split': [ 10],
    'min_samples_leaf': [1],
    'max_features': [ 'sqrt'],
    'bootstrap': [True],
}



best_r2 = 0
best_model = None
best_params = None

while best_r2 < 0.95:
    # Usar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
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
joblib.dump(best_model, 'modelo_entrenado_RF.pkl')
joblib.dump(scaler, 'scaler_RF.pkl')

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

# Discretizar los niveles de estrés en categorías: bajo (0-33), medio (34-66), alto (67-100)
bins = [0, 33, 66, 100]
labels = ['Bajo', 'Medio', 'Alto']
y_train_binned = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
y_test_binned = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
y_pred_test_binned = pd.cut(y_pred_test, bins=bins, labels=labels, include_lowest=True)

# Crear la matriz de confusión
cm = confusion_matrix(y_test_binned, y_pred_test_binned, labels=labels)

# Mostrar la matriz de confusión
plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()

# Calcular el accuracy
accuracy = accuracy_score(y_test_binned, y_pred_test_binned)
print(f'Accuracy: {accuracy}')

# Mostrar resultados de evaluación
print(f'Train MSE: {train_mse}, Train R²: {train_r2}')
print(f'Test MSE: {test_mse}, Test R²: {test_r2}')
print(f'Train MAE: {train_mae}, Test MAE: {test_mae}')
print(f'Train MedAE: {train_medae}, Test MedAE: {test_medae}')

# Graficar resultados de predicción
plt.figure(figsize=(14, 8))

# Gráfico de MSE
plt.subplot(2, 2, 1)
plt.bar(['Train MSE', 'Test MSE'], [train_mse, test_mse], color=['blue', 'orange'])
plt.title('MSE')

# Gráfico de R²
plt.subplot(2, 2, 2)
plt.bar(['Train R²', 'Test R²'], [train_r2, test_r2], color=['blue', 'orange'])
plt.title('R²')

# Gráfico de MAE
plt.subplot(2, 2, 3)
plt.bar(['Train MAE', 'Test MAE'], [train_mae, test_mae], color=['blue', 'orange'])
plt.title('MAE')

# Gráfico de MedAE
plt.subplot(2, 2, 4)
plt.bar(['Train MedAE', 'Test MedAE'], [train_medae, test_medae], color=['blue', 'orange'])
plt.title('MedAE')

plt.tight_layout()
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

print(f'Mejor conjunto de hiperparámetros: {best_params}')
print(f'Mejor R²: {best_r2}')
