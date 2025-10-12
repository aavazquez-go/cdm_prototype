import streamlit as st

# Inicializar
if "num_modelos" not in st.session_state:
    st.session_state.num_modelos = 1

def actualizar_num_modelos():
    st.session_state.num_modelos = st.session_state.input_num_modelos

with st.sidebar:
    st.title("Configuración")
    st.number_input(
        "Cantidad de modelos",
        min_value=1,
        max_value=100,
        value=st.session_state.num_modelos,
        step=1,
        key="input_num_modelos",  # clave para identificar el widget
        on_change=actualizar_num_modelos  # se llama cuando cambia
    )

st.write(f"Modelos configurados: {st.session_state.num_modelos}")