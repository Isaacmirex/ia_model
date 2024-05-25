import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Cargar los datos
file_path = './Data/datos_biometricos_lineales.xlsx'
data = pd.read_excel(file_path)

# Preparar los datos
X = data.drop(columns=['Estrés (%)'])
y = data['Estrés (%)']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo con los hiperparámetros especificados
model = RandomForestRegressor(
    bootstrap=True,
    ccp_alpha=0.0,
    criterion='squared_error',
    max_depth=10,
    max_features='sqrt',
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=10,
    min_weight_fraction_leaf=0.0,
    monotonic_cst=None,
    n_estimators=100,
    n_jobs=None,
    oob_score=False,
    random_state=42,
    verbose=0,
    warm_start=False
)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluar el modelo
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

# Mostrar resultados
print(f'Mejor conjunto de hiperparámetros: {model.get_params()}')
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
