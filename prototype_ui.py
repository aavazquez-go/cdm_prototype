import streamlit as st
import pandas as pd
from io import StringIO
from settings import Model, DeepSurvModel, LifelinesCoxPHModel, CoxCCModel, CoxTimeModel

# Inicializar
if "num_modelos" not in st.session_state:
    st.session_state.num_modelos = 3

if 'input_dataset' not in st.session_state:
    st.session_state.input_dataset = None

#if 'models' not in st.session_state:
st.session_state.models = []

def actualizar_num_modelos():
    st.session_state.num_modelos = st.session_state.input_num_modelos

def model_factory(model_type)-> Model| None:
    if model_type == "DeepSurv":
        return DeepSurvModel()
    elif model_type == "Lifelines-PHCox":
        return LifelinesCoxPHModel()
    elif model_type == "CoxCC":
        return CoxCCModel()
    elif model_type == "CoxTime":
        return CoxTimeModel()
    else:
        return None

# Configuracion visual del prototipo
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
    st.markdown("---")
    st.markdown("### Modelos para usar")
    model_types = st.multiselect(
        "Selecciona los tipos de modelos a usar",
        options=["Lifelines-PHCox", "DeepSurv", "CoxCC", "CoxTime"],
        default=["Lifelines-PHCox", "DeepSurv", "CoxCC", "CoxTime"]
    )

st.title("Prototipo")
st.write(f"Modelos a configurar: {st.session_state.num_modelos}")

st.markdown("## Cargar Dataset")

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

if st.session_state.input_dataset is not None:
    st.dataframe(st.session_state.input_dataset)

# Configuracion de los modelos
st.markdown("## Configuración de Modelos")
tabs_title = [f"Modelo {i+1}" for i in range(st.session_state.num_modelos)]
tabs = st.tabs(tabs_title)
for i, tab in enumerate(tabs):
    with tab:
        st.markdown(f"### Configuración del Modelo {i+1}")
        model_name = st.text_input(f"Nombre del Modelo {i+1}", key=f"model_name_{i}")
        model_type = st.selectbox(
            f"Tipo de Modelo {i+1}",
            model_types,
            key=f"model_type_{i}"
        )
        model = model_factory(model_type)
        if model is not None:
            model.set_name(model_name)
            if st.session_state.input_dataset is not None:
                model.set_input_df(st.session_state.input_dataset)
            model.show_ui()
        st.session_state.models.append(model)
        st.write("Detalles del modelo:")
        if model is not None:
            model.model_details()
        
        st.markdown("---")
        st.markdown("### Model prediction")
        if st.button(f"Predict Model {i+1}") and model is not None and st.session_state.input_dataset is not None:
            df_pred = model.predict(st.session_state.input_dataset)
            st.markdown(f"**{model.name}-{model.type}** prediction:")
            st.dataframe(df_pred)


st.markdown("## Realizar predicciones")

if st.button("Models Predict") and st.session_state.input_dataset is not None:
    models = st.session_state.models
    input_dataset = st.session_state.input_dataset
    for m in models:
        df_pred = m.predict(input_dataset)
        st.markdown(f"**{m.name}-{m.type}** prediction:")
        st.dataframe(df_pred)

st.markdown("## Toma de decisiones")