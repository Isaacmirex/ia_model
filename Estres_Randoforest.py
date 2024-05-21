import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

# Cargar datos
df_stress_percibido = pd.read_excel('Data/datos_estres_pss.xlsx')
df_biometricos = pd.read_excel('Data/datos_personas_simulados.xlsx')

# Preprocesar datos
df_stress_percibido['id'] = df_stress_percibido.index
df_biometricos['id'] = df_biometricos.index
df = pd.merge(df_stress_percibido, df_biometricos, on='id').drop(columns=['id']).dropna()

# Separar características y etiquetas
X = df.drop(columns=['Estrés (%)'])
y = df['Estrés (%)']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Definir parámetros para la búsqueda
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2']
}

# Inicializar el modelo
gbc = GradientBoostingClassifier(random_state=42)

# Realizar la búsqueda de hiperparámetros
random_search = RandomizedSearchCV(estimator=gbc, param_distributions=param_grid, 
                                   n_iter=100, cv=5, n_jobs=-1, verbose=2, scoring='accuracy', random_state=42)
random_search.fit(X_train_smote, y_train_smote)

# Evaluar el mejor modelo
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

# Binarizar las etiquetas
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# Calcular ROC AUC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], best_model.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Métricas
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Mejores Hiperparámetros:", random_search.best_params_)
print("Exactitud:", accuracy)
print("ROC AUC por clase:", roc_auc)
print("Reporte de Clasificación:\n", class_report)
print("Matriz de Confusión:\n", conf_matrix)

# Gráficos

# Matriz de confusión
fig, ax = plt.subplots(figsize=(10, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(ax=ax)
plt.title('Matriz de Confusión')
plt.show()

# Curva ROC Multiclase
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Clase {i} (área = {roc_auc[i]:.2f})')
    
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC Multiclase')
plt.legend(loc="lower right")
plt.show()

# Importancia de características
feature_importances = pd.DataFrame(best_model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(x=feature_importances.importance, y=feature_importances.index, ax=ax)
plt.title('Importancia de Características')
plt.show()