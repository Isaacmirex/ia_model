import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from scipy.stats import pearsonr

# Cargar los datos
datos_estres = pd.read_excel("data/datos_estres_pss.xlsx")
datos_personas = pd.read_excel("data/datos_personas_simulados.xlsx")

# Combinar los datos en un solo DataFrame
datos_combinados = pd.merge(datos_estres, datos_personas, left_index=True, right_index=True)

# Mostrar los primeros registros del DataFrame
print("Datos combinados:")
print(datos_combinados.head())

# Mostrar información estadística de los datos
print("\nInformación estadística de los datos:")
print(datos_combinados.describe())

# Visualizar la relación entre las características y el objetivo
print("\nVisualización de la relación entre las características y el objetivo:")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Resaltar en rojo las personas con un nivel de estrés por encima del 50%
rojo = datos_combinados['Estrés (%)'] > 50
azul = datos_combinados['Estrés (%)'] <= 50

axes[0, 0].scatter(datos_combinados['PSS'][azul], datos_combinados['Estrés (%)'][azul], c='blue', alpha=0.5)
axes[0, 0].scatter(datos_combinados['PSS'][rojo], datos_combinados['Estrés (%)'][rojo], c='red', alpha=0.5)
axes[0, 0].set_xlabel('PSS')
axes[0, 0].set_ylabel('Estrés (%)')
axes[0, 0].set_title('Relación entre PSS y Estrés (%)')

axes[0, 1].scatter(datos_combinados['Frecuencia Respiratoria'][azul], datos_combinados['Estrés (%)'][azul], c='blue', alpha=0.5)
axes[0, 1].scatter(datos_combinados['Frecuencia Respiratoria'][rojo], datos_combinados['Estrés (%)'][rojo], c='red', alpha=0.5)
axes[0, 1].set_xlabel('Frecuencia Respiratoria')
axes[0, 1].set_ylabel('Estrés (%)')
axes[0, 1].set_title('Relación entre Frecuencia Respiratoria y Estrés (%)')

axes[1, 0].scatter(datos_combinados['Frecuencia Cardíaca'][azul], datos_combinados['Estrés (%)'][azul], c='blue', alpha=0.5)
axes[1, 0].scatter(datos_combinados['Frecuencia Cardíaca'][rojo], datos_combinados['Estrés (%)'][rojo], c='red', alpha=0.5)
axes[1, 0].set_xlabel('Frecuencia Cardíaca')
axes[1, 0].set_ylabel('Estrés (%)')
axes[1, 0].set_title('Relación entre Frecuencia Cardíaca y Estrés (%)')

axes[1, 1].scatter(datos_combinados['Temperatura Corporal'][azul], datos_combinados['Estrés (%)'][azul], c='blue', alpha=0.5)
axes[1, 1].scatter(datos_combinados['Temperatura Corporal'][rojo], datos_combinados['Estrés (%)'][rojo], c='red', alpha=0.5)
axes[1, 1].set_xlabel('Temperatura Corporal')
axes[1, 1].set_ylabel('Estrés (%)')
axes[1, 1].set_title('Relación entre Temperatura Corporal y Estrés (%)')

plt.tight_layout()
plt.show()

# Definir las características (X) y el objetivo (y)
X = datos_combinados.drop(['Estrés (%)'], axis=1)  # Características
y = datos_combinados['Estrés (%)']  # Objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo SVM
modelo_svm = SVR(kernel='linear')
modelo_svm.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
predicciones = modelo_svm.predict(X_test)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)
mae = mean_absolute_error(y_test, predicciones)
mape = mean_absolute_percentage_error(y_test, predicciones)
medae = median_absolute_error(y_test, predicciones)
correlation = pearsonr(y_test, predicciones)[0]

print("\nMean Squared Error (MSE):", mse)
print("Coefficient of Determination (R^2):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Median Absolute Error (MedAE):", medae)
print("Pearson Correlation:", correlation)

# Visualizar las predicciones
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predicciones)
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.title('Valores reales vs Predicciones')
plt.show()
