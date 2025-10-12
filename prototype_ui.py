import streamlit as st
import pandas as pd
from io import StringIO

# Inicializar
if "num_modelos" not in st.session_state:
    st.session_state.num_modelos = 1

if 'input_dataset' not in st.session_state:
    st.session_state.input_dataset = None

def actualizar_num_modelos():
    st.session_state.num_modelos = st.session_state.input_num_modelos

with st.sidebar:
    st.title("Configuración")
    st.number_input(
        "Cantidad de modelos",
        min_value=1,
        max_value=10,
        value=st.session_state.num_modelos,
        step=1,
        key="input_num_modelos",  # clave para identificar el widget
        on_change=actualizar_num_modelos  # se llama cuando cambia
    )

st.title("Prototipo")
st.write(f"Modelos configurados: {st.session_state.num_modelos}")

# file with data to proccess
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    if uploaded_file.type == "text/csv":
        dataframe = pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        dataframe = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        dataframe = None
st.session_state.input_dataset = dataframe
if dataframe is not None:
    st.dataframe(dataframe)


tabs_title = [f"Modelo {i+1}" for i in range(st.session_state.num_modelos)]
tabs = st.tabs(tabs_title)
for i, tab in enumerate(tabs):
    with tab:
        st.header(f"Configuración del Modelo {i+1}")
        st.text_input(f"Nombre del Modelo {i+1}", key=f"model_name_{i}")
        st.selectbox(
            f"Tipo de Modelo {i+1}",
            ["Regresión", "Clasificación", "Clustering"],
            key=f"model_type_{i}"
        )
        st.slider(
            f"Hiperparámetro 1 del Modelo {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            key=f"hyperparam_1_{i}"
        )
        st.slider(
            f"Hiperparámetro 2 del Modelo {i+1}",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key=f"hyperparam_2_{i}"
        )
        st.write("Detalles del modelo:")
        st.json({
            "Nombre": st.session_state.get(f"model_name_{i}", ""),
            "Tipo": st.session_state.get(f"model_type_{i}", ""),
            "Hiperparámetro 1": st.session_state.get(f"hyperparam_1_{i}", 0.5),
            "Hiperparámetro 2": st.session_state.get(f"hyperparam_2_{i}", 10)
        })