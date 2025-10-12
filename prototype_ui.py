import streamlit as st
from settings import Models

# Inicializar el estado de sesión si no existe
if "num_modelos" not in st.session_state:
    st.session_state.num_modelos = 3  # valor por defecto

# Sidebar: input para la cantidad de modelos
st.sidebar.title("Configuración")
num_modelos_input = st.sidebar.number_input(
    "Cantidad de modelos",
    min_value=1,
    max_value=100,
    value=st.session_state.num_modelos,
    step=1
)

# Actualizar el estado de sesión si el valor cambia
if num_modelos_input != st.session_state.num_modelos:
    st.session_state.num_modelos = num_modelos_input
    st.experimental_rerun()  # Opcional: para reflejar cambios inmediatamente

# Ahora puedes usar st.session_state.num_modelos en cualquier parte del sistema
st.write(f"Actualmente estás manejando {st.session_state.num_modelos} modelo(s).")

# Ejemplo: generar inputs dinámicos según la cantidad de modelos
for i in range(st.session_state.num_modelos):
    st.text_input(f"Nombre del modelo {i+1}", key=f"modelo_{i}")