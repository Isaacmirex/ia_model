import pandas as pd
import random

# Generar datos aleatorios simulados
data = []
for _ in range(10000):
    estres = random.randint(10, 100)
    
    # Generar valores aleatorios simulados para cada sensor
    if estres < 95:
        fr = random.randint(12, 20)
        fc = random.randint(60, 100)
        tc = random.uniform(36.1, 37.2)
    else:
        fr = random.randint(20, 30)
        fc = random.randint(90, 120)
        tc = random.uniform(37.2, 38.5)
    
    data.append([estres, fr, fc, tc])

# Crear DataFrame
df = pd.DataFrame(data, columns=['Estrés (%)', 'Frecuencia Respiratoria', 'Frecuencia Cardíaca', 'Temperatura Corporal'])

# Guardar en un archivo Excel
df.to_excel('datos_personas_simulados.xlsx', index=False)