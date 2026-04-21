import pandas as pd
import joblib

MODELO_PATH = "modelo_knn.pkl"
SCALER_PATH = "scaler_knn.pkl"

try:
    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modelo cargado correctamente...\n")

except:
    print("Debe entrenar el modelo por primera vez...\n")
    exit()


m2 = float(input("metros_cuadrados: "))
num_habitaciones = float(input("numero de habitaciones: "))
num_banos = float(input("num_banos: "))
antiguedad = float(input("Antiguedad: "))
distancia_centro = float(input("distancia_centro: "))
estrato = float(input("estrato: "))
garaje = float(input("garaje: "))
zona = float(input("zona: "))

nuevo = pd.DataFrame(
    [[m2, num_habitaciones, num_banos,antiguedad,distancia_centro,estrato,garaje,zona]],
    columns=["metros_cuadrados", "num_habitaciones", "num_banos","antiguedad","distancia_centro","estrato","garaje","zona"]
)

nuevo = scaler.transform(nuevo)
pred = modelo.predict(nuevo)

print("\nValor de la casa:", round(pred[0], 2))
