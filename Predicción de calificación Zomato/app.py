import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title=Predicción de calificación Zomato, page_icon=🍽️, layout=wide)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, modelos entrenados)

@st.cache_resource
def load_models()
    return {
        Random Forest joblib.load(os.path.join(MODELS_DIR, modelo_rf.joblib)),
        KNN (k=10) joblib.load(os.path.join(MODELS_DIR, modelo_knn.joblib)),
        Regresión Lineal joblib.load(os.path.join(MODELS_DIR, modelo_lr.joblib)),
    }

models = load_models()

features = [
    online_order,
    book_table,
    cost,
    n_cuisines,
    log_votes,
    rest_type,
    location,
    type,
    city,
]

metrics = pd.DataFrame([
    {Modelo Random Forest, RMSE 0.1553, MAE 0.0869, R² 0.8738},
    {Modelo KNN (k=10), RMSE 0.3121, MAE 0.2257, R² 0.4903},
    {Modelo Regresión Lineal, RMSE 0.3280, MAE 0.2350, R² 0.4370},
    {Modelo Línea base, RMSE 0.4372, MAE 0.3486, R² -0.0005},
])

unique_values = {
    online_order [0, 1],
    book_table [0, 1],
    n_cuisines list(range(1, 10)),
    location list(range(0, 100)),
    rest_type list(range(0, 100)),
    type list(range(0, 20)),
    city list(range(0, 35)),
}

ranges = {
    cost (0.0, 10000.0),
    n_cuisines (1.0, 20.0),
    log_votes (0.0, 15.0),
    location (0.0, 500.0),
    rest_type (0.0, 500.0),
    type (0.0, 500.0),
    city (0.0, 500.0),
}

st.title(🍽️ Predicción de calificación de restaurantes — Zomato)
st.write(Aplicación de Machine Learning basada en los modelos entrenados en Google Colab.)
st.info(Las variables categóricas se muestran como códigos numéricos porque el archivo de entrenamiento contiene códigos y no incluye el diccionario que los traduce a nombres.)

tab1, tab2 = st.tabs([Realizar predicción, Comparación de modelos])

with tab1
    st.subheader(Ingresa las características del restaurante)
    values = {}
    left, right = st.columns(2)
    for i, col in enumerate(features)
        container = left if i % 2 == 0 else right
        with container
            opts = unique_values.get(col, [])
            if col in [online_order, book_table, n_cuisines, location, rest_type, type, city]
                if len(opts) = 100
                    values[col] = st.selectbox(f{col}, opts, help=Código numérico usado en el dataset original.)
                else
                    lo, hi = ranges[col]
                    values[col] = st.number_input(col, min_value=float(lo), max_value=float(hi), value=float(opts[0]))
            else
                lo, hi = ranges[col]
                default = float(pd.Series(opts).median()) if opts else float(lo)
                values[col] = st.number_input(col, min_value=float(lo), max_value=float(hi), value=default)

    selected_model = st.selectbox(
        Modelo para la predicción,
        list(models.keys()),
        index=list(models.keys()).index(Random Forest),
    )

    if st.button(Predecir calificación, type=primary)
        row = pd.DataFrame([[values[c] for c in features]], columns=features)
        raw_prediction = float(models[selected_model].predict(row)[0])
        prediction = float(np.clip(raw_prediction, 1.0, 5.0))

        st.success(fCalificación estimada (escala 1 a 5) {prediction.2f}  5.00)
        st.progress((prediction - 1.0)  4.0)
        st.caption(Es una estimación del modelo en escala de 1 a 5, no una calificación oficial del restaurante.)

with tab2
    st.subheader(Resultados de evaluación)
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    best = metrics.sort_values(RMSE).iloc[0]
    st.success(fSegún el RMSE del conjunto de prueba, el mejor modelo es {best['Modelo']}.)
    st.caption(RMSE y MAE menor es mejor. R² más alto es mejor. Las métricas corresponden a la evaluación realizada en el notebook.)