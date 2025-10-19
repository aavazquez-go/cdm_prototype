import streamlit as st
import pandas as pd
from settings import Model, DeepSurvModel, LifelinesCoxPHModel, CoxCCModel, CoxTimeModel
from assemblies import EnsembleManager

# Inicializar
if "num_modelos" not in st.session_state:
    st.session_state.num_modelos = 3

if 'input_dataset' not in st.session_state:
    st.session_state.input_dataset = None

# Inicializar resultados de predicción por modelo
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = {}

if 'models' not in st.session_state:
    st.session_state.models = []

def actualizar_num_modelos():
    st.session_state.num_modelos = st.session_state.input_num_modelos

def model_factory(model_type, tab_index = None)-> Model| None:
    if model_type == "DeepSurv":
        return DeepSurvModel(tab_index=tab_index)
    elif model_type == "Lifelines-CoxPH":
        return LifelinesCoxPHModel(tab_index=tab_index)
    elif model_type == "CoxCC":
        return CoxCCModel(tab_index=tab_index)
    elif model_type == "CoxTime":
        return CoxTimeModel(tab_index=tab_index)
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
        options=["Lifelines-CoxPH", "DeepSurv", "CoxCC", "CoxTime"],
        default=["Lifelines-CoxPH", "DeepSurv", "CoxCC", "CoxTime"]
    )
    st.markdown("---")
    st.markdown("### Modelos de ensamble")
    assembly_model = st.selectbox(
        "Selecciona el modelo de ensamble",
        ("Survival Function Averaging","Median Time Averaging","Stacking","Performance-weighted ensemble", "Rank-based ensemble","Bayesian Model Averaging","Voting/Ranking Consensus")
    )

st.title("Prototipo CADM para predicción de insolvencia en PYMEs")
st.write(f"Modelos a configurar: {st.session_state.num_modelos}")

st.markdown("## Cargar dataset para predecir")

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

# Configuración de los modelos
st.markdown("## Configuración de Modelos")

# Aseguramos que st.session_state.models tenga la longitud correcta
if len(st.session_state.models) != st.session_state.num_modelos:
    # Extender o recortar la lista de modelos
    current_len = len(st.session_state.models)
    target_len = st.session_state.num_modelos
    if target_len > current_len:
        # Añadir nuevos modelos (None por ahora)
        st.session_state.models.extend([None] * (target_len - current_len))
    elif target_len < current_len:
        # Recortar
        st.session_state.models = st.session_state.models[:target_len]

tabs_title = [f"Modelo {i+1}" for i in range(st.session_state.num_modelos)]
tabs = st.tabs(tabs_title)

for i, tab in enumerate(tabs):
    with tab:
        st.markdown(f"### Configuración del Modelo {i+1}")
        
        # Leer o crear el modelo para esta pestaña
        current_model = st.session_state.models[i]
        
        model_name = st.text_input(f"Nombre del Modelo {i+1}", value=current_model.name if current_model else "", key=f"model_name_{i}")
        model_type = st.selectbox(
            f"Tipo de Modelo {i+1}",
            model_types,
            index=model_types.index(current_model.type) if current_model and current_model.type in model_types else 0,
            key=f"model_type_{i}"
        )

        # Si el tipo cambió o no hay modelo, crear uno nuevo
        if current_model is None or current_model.type != model_type:
            new_model = model_factory(model_type,tab_index=i)
            if new_model:
                new_model.set_name(model_name)
                st.session_state.models[i] = new_model
            else:
                st.session_state.models[i] = None
                st.warning("Modelo no soportado.")
                continue
        else:
            # Actualizar nombre si cambió
            current_model.set_name(model_name)
            st.session_state.models[i] = current_model

        model = st.session_state.models[i]
        if model is not None:
            if st.session_state.input_dataset is not None:
                model.set_input_df(st.session_state.input_dataset)
            model.show_ui()
            st.write("Detalles del modelo:")
            model.model_details()

        st.markdown("---")
        st.markdown("### Model prediction")

        # Mostrar resultados anteriores si existen
        if i in st.session_state.prediction_results:
            res = st.session_state.prediction_results[i]
            st.markdown(f"**{res['name']}-{res['type']}** prediction:")
            st.dataframe(res['df_pred'])
            st.pyplot(res['survival_curve'])
            if res['df_median_time'] is not None:
                st.dataframe(res['df_median_time'])
            st.write(f"Indice de Concordancia del modelo = {res['c_index']}")

        # Botón de predicción
        if st.button(f"Predict Model {i+1}", key=f"predict_btn_{i}") and model is not None and st.session_state.input_dataset is not None:
            try:
                df_pred = model.predict()
                survival_curve = model.get_survival_curve()
                df_median_time = model.predict_median_survival_time()
                c_index = model.concordance_index()
                
                # Guardar en session_state
                st.session_state.prediction_results[i] = {
                    'name': model.name,
                    'type': model.type,
                    'df_pred': df_pred,
                    'survival_curve': survival_curve,
                    'df_median_time': df_median_time,
                    'c_index': c_index
                }

                # IMPORTANTE: Asegurar que los atributos del modelo se actualicen
                # model.surv = df_pred
                # model.surv_curve = survival_curve
                # model.median_time = df_median_time
                # model.c_index = c_index

                # # Actualizar el modelo en session_state
                # st.session_state.models[i] = model
                st.session_state.models[i].surv = df_pred
                st.session_state.models[i].surv_curve = survival_curve
                st.session_state.models[i].median_time = df_median_time
                st.session_state.models[i].c_index = c_index

                st.success(f"Predicción completada para {model.name}")
                st.rerun()

                # Streamlit se re-ejecutará y mostrará el resultado arriba
            except Exception as e:
                st.error(f"Error al predecir con el modelo {model.name}: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

st.markdown("## Realizar predicciones de Ensamble")
# Debug: mostrar estado de los modelos
with st.expander("🔍 Debug - Estado de modelos"):
    for i, m in enumerate(st.session_state.models):
        if m is not None:
            has_surv = hasattr(m, 'surv') and m.surv is not None
            st.write(f"Modelo {i+1} ({m.name}): surv={'✅' if has_surv else '❌'}")
        else:
            st.write(f"Modelo {i+1}: None")

if st.button("Generar Predicción de Ensamble"):
    # Verificar que hay modelos con predicciones
    models_with_predictions = [m for m in st.session_state.models if m is not None and hasattr(m, 'surv') and m.surv is not None]
    
    if not models_with_predictions:
        st.error("❗ Por favor, realiza predicciones en al menos un modelo antes de generar el ensamble.")
    else:
        st.info(f"Generando ensamble con {len(models_with_predictions)} modelos")

        try:
            ensemble_mgr = EnsembleManager(assembly_model)
            ensemble_mgr.fit(st.session_state.models)
            ensemble_surv, ensemble_median = ensemble_mgr.predict()
            
            st.markdown("### Predicción del Ensamble")
            st.markdown(f"#### {assembly_model}")
            st.dataframe(ensemble_surv)
            st.pyplot(ensemble_mgr.plot_ensemble_survival())
            st.dataframe(ensemble_median)
            
            info = ensemble_mgr.get_strategy_info()
            st.json(info)
            st.rerun()
        except Exception as e:
            st.error(f"Error al generar el ensamble: {e}")
            import traceback
            st.error(traceback.format_exc())