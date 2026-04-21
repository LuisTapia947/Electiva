from sklearn import pipeline
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODELO_PATH = "modelo_casas.pkl"

try:
    modelo = joblib.load(MODELO_PATH)
    print("Modelo cargado correctamente...\n")

except:
    print("Entrenando modelo por primera vez...\n")

    data = pd.read_csv("Electiva\Dataset_viviendas.csv")

    x = data[["metros_cuadrados", "num_habitaciones", "num_banos","antiguedad","distancia_centro","estrato","garaje","zona"]]
    y = data["valor_casa"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.4, random_state=42
    )

    modelo = Pipeline([("modelo", DecisionTreeRegressor(max_depth=5))])
    modelo.fit(x_train, y_train)

    y_pred = modelo.predict(x_test)

    print("Evaluación del modelo:")
    print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
    print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    print("R2:", round(r2_score(y_test, y_pred), 4))

    joblib.dump(modelo, MODELO_PATH)

    print("\nModelo guardado.\n")