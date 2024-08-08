import joblib
import numpy as np

def load_model_and_scaler(model_path='modelo_entrenado_RF.pkl', scaler_path='scaler_RF.pkl'):
    # Cargar el modelo y el escalador desde los archivos
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def get_user_input():
    # Solicitar al usuario que ingrese los valores
    frecuencia_cardiaca = float(input("Ingrese la Frecuencia Cardíaca: "))
    frecuencia_respiratoria = float(input("Ingrese la Frecuencia Respiratoria: "))
    temperatura_corporal = float(input("Ingrese la Temperatura Corporal: "))
    edad = float(input("Ingrese la Edad: "))
    
    # Crear un arreglo con los valores ingresados
    user_data = np.array([[frecuencia_cardiaca, frecuencia_respiratoria, temperatura_corporal, edad]])
    return user_data

def main():
    # Cargar el modelo y el escalador
    model, scaler = load_model_and_scaler()
    
    # Obtener los datos del usuario
    user_data = get_user_input()
    
    # Normalizar los datos del usuario
    user_data_scaled = scaler.transform(user_data)
    
    # Realizar la predicción
    stress_prediction = model.predict(user_data_scaled)
    
    # Mostrar la predicción
    print(f"Predicción de estrés: {stress_prediction[0]:.2f}%")

if __name__ == "__main__":
    main()
