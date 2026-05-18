import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# =========================
# CARGA DE MODELOS
# =========================
modelo_logistico = joblib.load("modelos/modelo_logistico.pkl")
modelo_tree = joblib.load("modelos/modelo_tree.pkl")

# =========================
# HEADER
# =========================
st.title("🚢 Titanic Survival Predictor")

st.markdown("### 👩‍🎓 Daniela Esther Paucar Chavez")
st.markdown("NRC: 6816")

st.markdown("### 📚 Colab (modo lectura)")
st.markdown("https://colab.research.google.com/drive/17LDmXjnCHvVJ2s43xYyd1kSb5_PZ78Ui?usp=sharing")

st.markdown("---")

# =========================
# INPUTS
# =========================
st.header("🧾 Datos del pasajero")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Pclass (Clase del pasajero)", [1, 2, 3])
    age = st.slider("Edad", 0, 80, 25)
    sibsp = st.number_input("SibSp (Número de hermanos o esposo/esposa que viajaban con el pasajero)", 0, 10, 0)

with col2:
    parch = st.number_input("Parch (Número de padres o hijos que viajaban con el pasajero)", 0, 10, 0)
    fare = st.number_input("Tarifa pagada", 0.0, 600.0, 50.0)
    sex = st.selectbox("Sexo", ["Male(Masculino)", "Female(Femenino)"])
    embarked = st.selectbox("Puerto de embarque", ["S", "C", "Q"])

# =========================
# SELECCIÓN DE MODELO
# =========================
modelo_elegido = st.selectbox(
    "Modelo de predicción",
    ["Logistic Regression", "Decision Tree"]
)

# =========================
# BOTÓN DE PREDICCIÓN
# =========================
if st.button("🔍 Predecir supervivencia"):

    # DataFrame con MISMAS columnas que Colab
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Sex": sex,
        "Embarked": embarked
    }])

    # MISMO procesamiento que entrenamiento
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Selección de modelo
    if modelo_elegido == "Logistic Regression":
        model = modelo_logistico
    else:
        model = modelo_tree

    # Alinear columnas con entrenamiento
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predicción
    pred = model.predict(input_df)[0]

    st.markdown("---")

    # Resultado
    if pred == 1:
        st.success("🎉 El pasajero SOBREVIVIRÍA")
        st.balloons()
    else:
        st.error("⚠️ El pasajero NO SOBREVIVIRÍA")
        st.snow()

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Modelo entrenado en Colab usando dataset Titanic + Logistic Regression y Decision Tree")
