import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configurar página
st.set_page_config(page_title="Modelos de Predicción", layout="wide")
st.title("Modelos de Predicción")

# Rutas de modelos
MODELO_ARBOL_PATH = "modelo_casas.pkl"
MODELO_KNN_PATH = "modelo_knn.pkl"
MODELO_SCALER_PATH = "scaler_knn.pkl"
MODELO_REGRESION_PATH = "modelo_regresion.pkl"
DATASET_PATH = "Dataset_viviendas.csv"

# Función para entrenar modelos si no existen
def entrenar_modelos():
    """Entrena todos los modelos si no existen"""
    if not all([os.path.exists(MODELO_ARBOL_PATH), 
                os.path.exists(MODELO_KNN_PATH),
                os.path.exists(MODELO_REGRESION_PATH)]):
        
        st.info("Entrenando modelos por primera vez...")
        
        # Cargar datos
        data = pd.read_csv(DATASET_PATH)
        
        # Entrenamiento Árbol de Decisiones
        if not os.path.exists(MODELO_ARBOL_PATH):
            x = data[["metros_cuadrados", "num_habitaciones", "num_banos", "antiguedad", 
                     "distancia_centro", "estrato", "garaje", "zona"]]
            y = data["valor_casa"]
            x_train, _, y_train, _ = train_test_split(x, y, test_size=0.4, random_state=42)
            
            modelo_arbol = DecisionTreeRegressor(max_depth=5)
            modelo_arbol.fit(x_train, y_train)
            joblib.dump(modelo_arbol, MODELO_ARBOL_PATH)
        
        # Entrenamiento KNN
        if not os.path.exists(MODELO_KNN_PATH):
            x = data[["metros_cuadrados", "num_habitaciones", "num_banos", "antiguedad", 
                     "distancia_centro", "estrato", "garaje", "zona"]]
            y = data["valor_casa"]
            x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            
            modelo_knn = KNeighborsClassifier(n_neighbors=5)
            modelo_knn.fit(x_train, y_train)
            
            joblib.dump(modelo_knn, MODELO_KNN_PATH)
            joblib.dump(scaler, MODELO_SCALER_PATH)
        
        # Entrenamiento Regresión Lineal
        if not os.path.exists(MODELO_REGRESION_PATH):
            x = data[["metros_cuadrados", "num_habitaciones", "num_banos", "antiguedad",
                     "distancia_centro", "estrato", "garaje", "zona"]]
            y = data["valor_casa"]
            x_train, _, y_train, _ = train_test_split(x, y, test_size=0.4, random_state=42)
            
            modelo_regresion = LinearRegression()
            modelo_regresion.fit(x_train, y_train)
            joblib.dump(modelo_regresion, MODELO_REGRESION_PATH)
        
        st.success("Modelos entrenados correctamente!")

# Entrenar modelos si no existen
entrenar_modelos()

# Cargar modelos
modelo_arbol = joblib.load(MODELO_ARBOL_PATH)
modelo_knn = joblib.load(MODELO_KNN_PATH)
scaler = joblib.load(MODELO_SCALER_PATH)
modelo_regresion = joblib.load(MODELO_REGRESION_PATH)

# Seleccionar modelo
st.sidebar.title("Seleccionar Modelo")
modelo = st.sidebar.radio("Elige un modelo:", 
                          ["Árbol de Decisión", "KNN Regresión", "Regresión Lineal"])

if modelo == "Árbol de Decisión":
    st.header("Árbol de Decisión - Predicción de Valor de Casa")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        m2 = st.number_input("Metros cuadrados:", min_value=0.0, value=100.0)
        num_habitaciones = st.number_input("Número de habitaciones:", min_value=0, value=3)
    with col2:
        num_banos = st.number_input("Número de baños:", min_value=0, value=2)
        antiguedad = st.number_input("Antigüedad (años):", min_value=0, value=10)
    with col3:
        distancia_centro = st.number_input("Distancia al centro (km):", min_value=0.0, value=5.0)
        estrato = st.number_input("Estrato:", min_value=1, max_value=6, value=3)
    with col4:
        garaje = st.number_input("Garaje:", min_value=0, value=1)
        zona = st.number_input("Zona:", min_value=0, value=1)
    
    if st.button("Predecir valor de casa", key="arbol"):
        nuevo = pd.DataFrame(
            [[m2, num_habitaciones, num_banos, antiguedad, distancia_centro, estrato, garaje, zona]],
            columns=["metros_cuadrados", "num_habitaciones", "num_banos", "antiguedad",
                    "distancia_centro", "estrato", "garaje", "zona"]
        )
        pred = modelo_arbol.predict(nuevo)
        st.success(f" Valor predicho de la casa: ${pred[0]:,.2f}")

elif modelo == "KNN Regresión":
    st.header(" KNN - Predicción de Valor de Casa")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        m2 = st.number_input("Metros cuadrados:", min_value=0.0, value=100.0, key="knn_m2")
        num_habitaciones = st.number_input("Número de habitaciones:", min_value=0, value=3, key="knn_hab")
    with col2:
        num_banos = st.number_input("Número de baños:", min_value=0, value=2, key="knn_ban")
        antiguedad = st.number_input("Antigüedad (años):", min_value=0, value=10, key="knn_ant")
    with col3:
        distancia_centro = st.number_input("Distancia al centro (km):", min_value=0.0, value=5.0, key="knn_dist")
        estrato = st.number_input("Estrato:", min_value=1, max_value=6, value=3, key="knn_est")
    with col4:
        garaje = st.number_input("Garaje:", min_value=0, value=1, key="knn_gar")
        zona = st.number_input("Zona:", min_value=0, value=1, key="knn_zon")
    
    if st.button("Predecir valor de casa", key="knn"):
        nuevo = pd.DataFrame(
            [[m2, num_habitaciones, num_banos, antiguedad, distancia_centro, estrato, garaje, zona]],
            columns=["metros_cuadrados", "num_habitaciones", "num_banos", "antiguedad",
                    "distancia_centro", "estrato", "garaje", "zona"]
        )
        nuevo_escalado = scaler.transform(nuevo)
        pred = modelo_knn.predict(nuevo_escalado)
        st.success(f" Valor predicho de la casa: ${pred[0]:,.2f}")

elif modelo == "Regresión Lineal":
    st.header(" Regresión Lineal - Predicción de Valor de Casa")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        m2 = st.number_input("Metros cuadrados:", min_value=0.0, value=100.0, key="reg_m2")
        num_habitaciones = st.number_input("Número de habitaciones:", min_value=0, value=3, key="reg_hab")
    with col2:
        num_banos = st.number_input("Número de baños:", min_value=0, value=2, key="reg_ban")
        antiguedad = st.number_input("Antigüedad (años):", min_value=0, value=10, key="reg_ant")
    with col3:
        distancia_centro = st.number_input("Distancia al centro (km):", min_value=0.0, value=5.0, key="reg_dist")
        estrato = st.number_input("Estrato:", min_value=1, max_value=6, value=3, key="reg_est")
    with col4:
        garaje = st.number_input("Garaje:", min_value=0, value=1, key="reg_gar")
        zona = st.number_input("Zona:", min_value=0, value=1, key="reg_zon")
    
    if st.button("Predecir valor de casa", key="regresion"):
        nuevo = pd.DataFrame(
            [[m2, num_habitaciones, num_banos, antiguedad, distancia_centro, estrato, garaje, zona]],
            columns=["metros_cuadrados", "num_habitaciones", "num_banos", "antiguedad",
                    "distancia_centro", "estrato", "garaje", "zona"]
        )
        pred = modelo_regresion.predict(nuevo)
        st.success(f"Valor predicho de la casa: ${pred[0]:,.2f}")

st.divider()
st.info("📊 Aplicación desarrollada con Streamlit Community Cloud")
