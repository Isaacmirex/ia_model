import pandas as pd
import numpy as np

# Definir la función para generar los datos simulados de una persona
def generar_persona():
    persona = {}
    for i in range(1, 15):
        respuesta = np.random.randint(0, 5)  # Generar un número aleatorio entre 0 y 4
        persona[f"Pregunta_{i}"] = respuesta
    return persona

# Generar datos para 200 personas
datos_personas = [generar_persona() for _ in range(2000)]

# Crear un DataFrame de Pandas con los datos
df = pd.DataFrame(datos_personas)

# Renombrar las columnas para que coincidan con las preguntas de la escala PSS
columnas = {
    "Pregunta_1": "P_1",
    "Pregunta_2": "P_2",
    "Pregunta_3": "P_3",
    "Pregunta_4": "P_4",
    "Pregunta_5": "P_5",
    "Pregunta_6": "P_6",
    "Pregunta_7": "P_7",
    "Pregunta_8": "P_8",
    "Pregunta_9": "P_9",
    "Pregunta_10": "P_10",
    "Pregunta_11": "P_11",
    "Pregunta_12": "P_12",
    "Pregunta_13": "P_13",
    "Pregunta_14": "P_14"
}
df = df.rename(columns=columnas)

# Calcular el puntaje de estrés percibido (PSS) para cada persona
columnas_a_invertir = ['P_4', 'P_5', 'P_6', 'P_7', 'P_9', 'P_10', 'P_13']
df['PSS'] = 28 - df[columnas_a_invertir].sum(axis=1)

# Reordenar las columnas para que la columna de estrés percibido esté al principio
column_order = ['PSS'] + [f"P_{i}" for i in range(1, 15)]
df = df[column_order]

# Exportar los datos a un archivo Excel
df.to_excel("datos_estres_pss.xlsx", index=False)
