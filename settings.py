import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pandas import DataFrame
import pickle
from lifelines.utils import concordance_index


import torch
import torchtuples as tt
import joblib

import os
import tempfile

# Redirigir la carpeta de datos de pycox a /tmp (escribible en Streamlit Cloud)
os.environ["PYCOX_DATA_DIR"] = os.path.join(tempfile.gettempdir(), "pycox_data")

from pycox.models import CoxPH, CoxCC, CoxTime
from pycox.evaluation import EvalSurv
from pycox.models.cox_time import MLPVanillaCoxTime

from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self.name = ""
        self.type = ""
        self.parameters = {}
        self.input_df = None
        self.median_time = None
        self.surv = None
        self.surv_curve = None
        self.c_index = None
    
    def set_name(self, name):
        self.name = name
    
    def set_type(self, model_type):
        self.type = model_type
    
    def set_parameters(self, parameters):
        self.parameters = parameters
    
    def set_input_df(self, df):
        self.input_df = df
    
    @abstractmethod
    def get_survival_func(self):
        pass

    @abstractmethod
    def show_ui(self)->None:
        pass

    @abstractmethod
    def predict(self) -> DataFrame:
        pass

    @abstractmethod
    def model_details(self)-> None:
        pass

    @abstractmethod
    def get_survival_curve(self):
        pass

    @abstractmethod
    def predict_median_survival_time(self) -> DataFrame:
        pass

    @abstractmethod
    def concordance_index(self)-> float | None:
        pass

class DeepSurvModel(Model):
    instance_count = 0
    def __init__(self, tab_index = None):
        super().__init__()
        DeepSurvModel.instance_count += 1
        self.type = "DeepSurv"
        self.tab_index = tab_index
        self.instance_id = f"deepsurv_tab_{tab_index}" if tab_index is not None else "deepsurv_temp"
        self.net  = tt.practical.MLPVanilla( #type: ignore
            in_features=45, 
            num_nodes=[32, 32], 
            out_features = 1, 
            batch_norm=True, 
            dropout=0.1, 
            output_bias=False
        )
        self.preprocessing_file = None
        self.state_dict_file = None
        self.baseline_hazards_file = None
        self.evaluation_file = None
        self.model = None
        self.model_loaded = None
        self.preprocessed_input = None
        self.evaluation = None
    
    def show_ui(self):
        st.markdown("#### DeepSurv")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Cargue el arhivo de preprocesamiento del modelo", 
            accept_multiple_files=False, 
            type="pkl", 
            key=f"deepsurv_preprocessing_file_uploader_{self.instance_id}"
        )
        if preprocessing_file:
            st.session_state[f"{self.instance_id}_preprocessing_file"] = preprocessing_file
            # st.success("✅ Preprocessing file uploaded successfully.")
        elif st.session_state.get(f"{self.instance_id}_preprocessing_file", None):
            st.info(f"Archivo cargado: {st.session_state[f'{self.instance_id}_preprocessing_file'].name}")
        self.preprocessing_file = st.session_state.get(f"{self.instance_id}_preprocessing_file", None)

        # State dict file
        state_dict_file = st.file_uploader(
            "Cargue el archivo de los pesos del modelo", 
            accept_multiple_files=False, 
            type="pt", 
            key=f"deepsurv_state_dict_file_uploader_{self.instance_id}"
        )
        if state_dict_file:
            st.session_state[f"{self.instance_id}_state_dict_file"] = state_dict_file
        self.state_dict_file = st.session_state.get(f"{self.instance_id}_state_dict_file", None)

        # Baseline hazards file
        baseline_hazards_file = st.file_uploader(
            "Cargue el archivo de linea base de riesgos", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"deepsurv_baseline_hazards_file_uploader_{self.instance_id}"
        )
        if baseline_hazards_file:
            st.session_state[f"{self.instance_id}_baseline_hazards_file"] = baseline_hazards_file
        self.baseline_hazards_file = st.session_state.get(f"{self.instance_id}_baseline_hazards_file", None)

        # Evaluation file
        evaluation_file = st.file_uploader(
            "Cargue el archivo de evaluación del modelo", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"deepsurv_evaluation_file_uploader_{self.instance_id}"
        )
        if evaluation_file:
            st.session_state[f"{self.instance_id}_evaluation_file"] = evaluation_file
        self.evaluation_file = st.session_state.get(f"{self.instance_id}_evaluation_file", None)
    
    def predict(self)->DataFrame:
        isReady = True
        if self.state_dict_file is None:
            st.warning("❗ Por favor, cargue el archivo de pesos del modelo para continuar.")
            isReady = False
        if self.preprocessing_file is None:
            st.warning("❗ Por favor, cargue el archivo de preprocesamiento para continuar.")
            isReady = False
        if self.baseline_hazards_file is None:
            st.warning("❗ Por favor, cargue el archivo de linea base de riesgos para continuar.")
            isReady = False
        if self.evaluation_file is None:
            st.warning("❗ Por favor, cargue el archivo de evaluacion para continuar.")
            isReady = False
        if self.input_df is None:
            st.warning("❗ Por favor, cargue el archivo con el conjunto de datos de entrada para continuar.")
            isReady = False
        if not isReady:
            return pd.DataFrame()

        net = self.net
        state_dict = torch.load(self.state_dict_file, weights_only=True) #type: ignore
        net.load_state_dict(state_dict)
        model_loaded = CoxPH(net)
        # cargando el baseline hazards 
        loaded_baseline_hazards = joblib.load(self.baseline_hazards_file)
        model_loaded.baseline_hazards_ = loaded_baseline_hazards
        model_loaded.baseline_cumulative_hazards_ = model_loaded.compute_baseline_cumulative_hazards(
            set_hazards=False, 
            baseline_hazards_=loaded_baseline_hazards
        )
        self.model_loaded = model_loaded
        input_df = self.input_df
        preprocessor = joblib.load(self.preprocessing_file)
        X = preprocessor.transform(input_df).astype('float32')
        self.preprocessed_input = pd.DataFrame(X, columns=input_df.columns) #type: ignore
        surv = model_loaded.predict_surv_df(X)
        self.surv = surv
        # creando curva de supervivencia
        self.surv_curve = self._create_survival_curve(surv, model_loaded)
        # creando objeto de evaluacion
        self.evaluation = joblib.load(self.evaluation_file)
        return surv
    
    def _create_survival_curve(self, surv, model_loaded):
        baseline_survival = np.exp(-model_loaded.baseline_cumulative_hazards_)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()
        # Línea dashed para la baseline
        ax.step(
            baseline_survival.index,
            baseline_survival.values,
            where='post',
            label='Baseline (población promedio)',
            color='black',
            linestyle='--',
            linewidth=2
        )
        # graficar cada curva de supervivencia
        for i in range(surv.shape[1]):
            ax.step(surv.index, surv.iloc[:, i],where='post',label=f'Empresa {i+1}')
        
        plt.ylabel('Probabilidad de supervivencia')
        plt.xlabel('Tiempo')
        plt.title('Curvas de supervivencia predichas (modelo DeepSurv)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)
        return fig

    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "Type": self.type,
            "Preprocessing File": self.preprocessing_file.name if self.preprocessing_file else "Not uploaded",
            "State Dict File": self.state_dict_file.name if self.state_dict_file else "Not uploaded",
            "Baseline Hazards File": self.baseline_hazards_file.name if self.baseline_hazards_file else "Not uploaded",
            "Net": self.net
        })
    
    def get_survival_curve(self):
        return self.surv_curve
    
    def predict_median_survival_time(self) -> DataFrame:
        if self.surv is None or self.preprocessed_input is None:
            st.warning("❗ Please run prediction first to get median survival.")
            return pd.DataFrame()
        emp = {}
        for i in range(self.preprocessed_input.shape[0]):
            median_time = self._predict_median_survival_time(self.surv[[i]])
            emp[f"Caso {i+1}"] = median_time
        self.median_time = pd.DataFrame.from_dict(emp, orient='index', columns=['Median Survival Time'])
        return self.median_time
        
    def _predict_median_survival_time(self, surv):
        if self.surv is None:
            st.warning("❗ Please run prediction first to get median survival time.")
            return np.inf
        times = surv.index.values
        s = surv.iloc[:, 0].values
        idx = np.where(s <= 0.5)[0]
        return times[idx[0]] if len(idx) > 0 else np.inf

    def concordance_index(self)-> float | None:
        if self.evaluation is None:
            if self.evaluation_file is None:
                st.warning("Por favor, cargue el archivo de evaluación para continuar.")
                return None
            else:
                self.evaluation = joblib.load(self.evaluation_file)
        self.c_index = self.evaluation.concordance_td()#type:ignore
        return self.c_index

    def get_survival_func(self):
        return self.surv
    
class LifelinesCoxPHModel(Model):
    
    def __init__(self, tab_index = None):
        super().__init__()
        self.type = "Lifelines-CoxPH"
        self.instance_id = f"lifelines_coxph_tab_{tab_index}" if tab_index is not None else "lifelines_coxph_temp"
        self.curv = None
        self.model_file = None
        self.model_loaded = None
        self.preprocessing_file = None
        self.preprocessed_input = None        
    
    def show_ui(self):
        st.markdown("### Lifelines CoxPHFitter")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Cargar archivo de preprocesamiento del modelo", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"lifelines_coxph_preprocessing_file_uploader_{self.instance_id}"
        )
        if preprocessing_file:
            st.session_state[f"{self.instance_id}_preprocessing_file"] = preprocessing_file
            # st.success("✅ Preprocessing file uploaded successfully.")
        elif st.session_state.get(f"{self.instance_id}_preprocessing_file", None):
            st.info(f"Archivo cargado: {st.session_state[f'{self.instance_id}_preprocessing_file'].name}")
        self.preprocessing_file = st.session_state.get(f"{self.instance_id}_preprocessing_file", None)

        model_file = st.file_uploader(
            "Cargar archivo del modelo entrenado de CoxPHFitter",
            accept_multiple_files=False,
            type="joblib",
            key=f"lifelines_coxph_model_uploader_{self.instance_id}"
        )
        if model_file:
            st.session_state[f"{self.instance_id}_model_file"] = model_file
        elif st.session_state.get(f"{self.instance_id}_model_file", None):
            st.info(f"Archivo cargado: {st.session_state[f'{self.instance_id}_model_file'].name}")
        self.model_file = st.session_state.get(f"{self.instance_id}_model_file", None)
    
    def predict(self) -> pd.DataFrame:
        if self.input_df is None:
            st.warning("❗ Por favor, cargue el archivo del conjunto de datos para continuar.")
            return pd.DataFrame()
        if self.model_file is None:
            st.warning("❗ Por favor, cargue el archivo del modelo CoxPHFitter entrenado para continuar.")
            return pd.DataFrame()
        if self.preprocessing_file is None:
            st.warning("❗ Por favor, suba el archivo de preprocesamiento para continuar")
            return pd.DataFrame()
        preprocessor = joblib.load(self.preprocessing_file)
        model_loaded = joblib.load(self.model_file)
        self.model_loaded = model_loaded
        x_to_predict = preprocessor.fit_transform(self.input_df).astype('float32')
        new_to_predict_df = pd.DataFrame(x_to_predict, columns=self.input_df.columns)
        self.preprocessed_input = new_to_predict_df
        surv = model_loaded.predict_survival_function(new_to_predict_df)
        self.surv = surv
        self.surv_curve = self._create_survival_curve(surv, model_loaded)
        return surv
    
    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "Type": self.type,
            "Preprocessing File": self.preprocessing_file.name if self.preprocessing_file else "Not uploaded",
            "Model File": self.model_file.name if self.model_file else "Not uploaded"
        })
    
    def _create_survival_curve(self, surv, model_loaded):
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()
        # Línea dashed para la baseline
        baseline_survival = model_loaded.baseline_survival_
        ax.step(
            baseline_survival.index,
            baseline_survival.values,
            where='post',
            label='Baseline (población promedio)',
            color='black',
            linestyle='--',
            linewidth=2
        )
        # graficar cada curva de supervivencia
        for i in range(surv.shape[1]):
            ax.step(surv.index, surv.iloc[:, i],where='post',label=f'Empresa {i+1}')
        
        plt.ylabel('Probabilidad de supervivencia')
        plt.xlabel('Tiempo')
        plt.title('Curvas de supervivencia predichas (modelo Lifelines CoxPH)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)
        return fig    
    
    def get_survival_curve(self):
        if self.surv_curve is None:
            st.warning("❗ Please run prediction first to get survival curves.")
            return None
        return self.surv_curve
    
    def predict_median_survival_time(self) -> DataFrame:
        if self.surv is None or self.model_loaded is None or self.preprocessed_input is None:
            st.warning("❗ Por favor, ejecute primero la prediccion para obtener la mediana del tiempo de supervivencia.")
            return pd.DataFrame()
        
        # Obtener las medianas del modelo
        median_result = self.model_loaded.predict_median(self.preprocessed_input)
        
        # Normalizar el índice a "Caso X" format
        emp = {}
        for i, (idx, value) in enumerate(median_result.items()):
            emp[f"Caso {i+1}"] = value
        
        self.median_time = pd.DataFrame.from_dict(emp, orient='index', columns=['Median Survival Time'])
        return self.median_time

    def concordance_index(self)-> float | None:
        if self.model_loaded is None:
            st.warning("Por favor, debe ejecutar previamente la predicción.")
            return None

        self.c_index = float(self.model_loaded.concordance_index_)
        return self.c_index

    def get_survival_func(self):
        return self.surv
    
class CoxCCModel(Model):
    
    def __init__(self, tab_index = None):
        super().__init__()
        self.type = "CoxCC"
        self.tab_index = tab_index
        self.instance_id = f"coxcc_tab_{tab_index}" if tab_index is not None else "coxcc_temp"
        self.surv_curve = None
        self.loaded_preprocessor = None
        self.loaded_baseline_hazards = None
        self.loaded_net_state_dict = None
        self.net = tt.practical.MLPVanilla( #type: ignore
            in_features=45, 
            num_nodes=[32, 32], 
            out_features = 1, 
            batch_norm=True, 
            dropout=0.1, 
            output_bias=False
        )
        self.preprocessed_input = None
        self.evaluation_file = None
        self.evaluation = None
        self.median_time = None
    
    def show_ui(self):
        st.markdown("### CoxCC ")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Upload preprocessing model file", 
            accept_multiple_files=False, 
            type="pkl", 
            key=f"coxcc_preprocessing_file_uploader_{self.instance_id}"
        )
        if preprocessing_file:
            st.session_state[f"{self.instance_id}_preprocessing_file"] = preprocessing_file
            # st.success("✅ Preprocessing file uploaded successfully.")
        elif st.session_state.get(f"{self.instance_id}_preprocessing_file", None):
            st.info(f"Archivo cargado: {st.session_state[f'{self.instance_id}_preprocessing_file'].name}")
        self.preprocessing_file = st.session_state.get(f"{self.instance_id}_preprocessing_file", None)

        # State dict file
        state_dict_file = st.file_uploader(
            "Upload PyTorch weight models", 
            accept_multiple_files=False, 
            type="pt", 
            key=f"coxcc_state_dict_file_uploader_{self.instance_id}"
        )
        if state_dict_file:
            st.session_state[f"{self.instance_id}_state_dict_file"] = state_dict_file
        self.state_dict_file = st.session_state.get(f"{self.instance_id}_state_dict_file", None)

        # Baseline hazards file
        baseline_hazards_file = st.file_uploader(
            "Upload baseline hazards file", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"coxcc_baseline_hazards_file_uploader_{self.instance_id}"
        )
        if baseline_hazards_file:
            st.session_state[f"{self.instance_id}_baseline_hazards_file"] = baseline_hazards_file
        self.baseline_hazards_file = st.session_state.get(f"{self.instance_id}_baseline_hazards_file", None)

        #Model Evaluation file
        evaluation_file = st.file_uploader(
            "Cargar archivo de evaluacion del modelo",
            accept_multiple_files=False,
            type="joblib",
            key=f"coxcc_evaluation_file_uploader_{self.instance_id}"
        )
        if evaluation_file:
            st.session_state[f"{self.instance_id}_evaluation_file"] = evaluation_file
        self.evaluation_file = st.session_state.get(f"{self.instance_id}_evaluation_file", None)

    def predict(self) -> pd.DataFrame:
        isReady = True
        if self.state_dict_file is None:
            st.warning("❗ Por favor, suba el archivo de pesos del modelo para continuar.")
            isReady = False
        if self.preprocessing_file is None:
            st.warning("❗ Por favor, suba el archivo de preprocesamiento para continuar.")
            isReady = False
        if self.baseline_hazards_file is None:
            st.warning("❗ Por favor, suba el archivo de linea base de riesgos para continuar.")
            isReady = False
        if self.evaluation_file is None:
            st.warning("❗ Por favor, cargue el archivo de evaluacion para continuar.")
            isReady = False
        if self.input_df is None:
            st.warning("❗ Por favor, cargue el archivo del conjunto de datos para continuar.")
            isReady = False
        if not isReady:
            return pd.DataFrame()
        
        loaded_preprocessor = joblib.load(self.preprocessing_file)
        loaded_baseline_hazards = joblib.load(self.baseline_hazards_file)
        loaded_model_net_state_dict = torch.load(self.state_dict_file)

        net = self.net
        net.load_state_dict(loaded_model_net_state_dict)
        model_loaded = CoxCC(net, tt.optim.Adam) # type: ignore
        
        # cargando el baseline hazards
        model_loaded.baseline_hazards_ = loaded_baseline_hazards
        model_loaded.baseline_cumulative_hazards_ = model_loaded.compute_baseline_cumulative_hazards(
            set_hazards=False, 
            baseline_hazards_= loaded_baseline_hazards
        )
        
        X_preprocessed = loaded_preprocessor.transform(self.input_df).astype('float32')
        # X_preprocessed_df = pd.DataFrame(X_preprocessed, columns = self.input_df.columns)
        self.preprocessed_input = pd.DataFrame(X_preprocessed, columns = self.input_df.columns) # type: ignore

        surv = model_loaded.predict_surv_df(X_preprocessed)
        # print(f"Predicted survival values: {surv}")
        self.surv = surv
        self.surv_curve = self._create_survival_curve(surv, model_loaded)
        # creando objetos de evaluacion
        self.evaluation = joblib.load(self.evaluation_file)
        return surv

    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "Type": self.type,
            "Preprocessing File": self.preprocessing_file.name if self.preprocessing_file else "Not uploaded",
            "State Dict File": self.state_dict_file.name if self.state_dict_file else "Not uploaded",
            "Baseline Hazards File": self.baseline_hazards_file.name if self.baseline_hazards_file else "Not uploaded",
            "Net": self.net
        })
    
    def _create_survival_curve(self, surv, model_loaded):
        baseline_survival = np.exp(-model_loaded.baseline_cumulative_hazards_)
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()
        # Línea dashed para la baseline
        ax.step(
            baseline_survival.index,
            baseline_survival.values,
            where='post',
            label='Baseline (población promedio)',
            color='black',
            linestyle='--',
            linewidth=2
        )
        # graficar cada curva de supervivencia
        for i in range(surv.shape[1]):
            ax.step(surv.index, surv.iloc[:, i],where='post',label=f'Empresa {i+1}')
        
        plt.ylabel('Probabilidad de supervivencia')
        plt.xlabel('Tiempo')
        plt.title('Curvas de supervivencia predichas (modelo CoxCC)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)
        return fig

    def get_survival_curve(self):
        return self.surv_curve
    
    def predict_median_survival_time(self) -> DataFrame:
        if self.surv is None or self.preprocessed_input is None:
            st.warning("❗ Please run prediction first to get median survival.")
            return pd.DataFrame()
        emp = {}
        for i in range(self.preprocessed_input.shape[0]):
            median_time = self._predict_median_survival_time(self.surv[[i]])
            emp[f"Caso {i+1}"] = median_time
        
        self.median_time = pd.DataFrame.from_dict(emp, orient='index', columns=['Median Survival Time'])
        return self.median_time

    def _predict_median_survival_time(self, surv):
        if self.surv is None:
            st.warning("❗ Please run prediction first to get median survival time.")
            return np.inf
        times = surv.index.values
        s = surv.iloc[:, 0].values
        idx = np.where(s <= 0.5)[0]
        return times[idx[0]] if len(idx) > 0 else np.inf

    def concordance_index(self)-> float | None:
        if self.evaluation is None:
            if self.evaluation_file is None:
                st.warning("Por favor, cargue el archivo de evaluación para continuar.")
                return None
            else:
                self.evaluation = joblib.load(self.evaluation_file)
        self.c_index = float(self.evaluation.concordance_td())#type:ignore
        return self.c_index

    def get_survival_func(self):
        return self.surv
    
class CoxTimeModel(Model):
    
    def __init__(self, tab_index = None):
        super().__init__()
        self.type = "CoxTime"
        self.instance_id = f"coxtime_tab_{tab_index}" if tab_index is not None else "coxtime_temp"
        self.preprocessing_file = None
        self.state_dict_file = None
        self.baseline_hazards_file = None
        # self.net_file = None
        self.labtrans_file = None
        self.evaluation_file = None
        self.loaded_model = None
        self.net = MLPVanillaCoxTime(
            in_features = 45, 
            num_nodes = [32, 32], 
            batch_norm = True, 
            dropout = 0.1
        )
        self.state_dict = None
        self.loaded_baseline_hazards = None
        self.loaded_preprocessor = None
        self.loaded_labtrans = None
        self.evaluation = None
    
    def show_ui(self):
        st.markdown("### CoxTime")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Cargar el archivo con el modelo de preprocesamiento", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"coxtime_preprocessing_file_uploader_{self.instance_id}"
        )
        if preprocessing_file:
            st.session_state[f"{self.instance_id}_preprocessing_file"] = preprocessing_file
            # st.success("✅ Preprocessing file uploaded successfully.")
        elif st.session_state.get(f"{self.instance_id}_preprocessing_file", None):
            st.info(f"Archivo cargado: {st.session_state[f'{self.instance_id}_preprocessing_file'].name}")
        self.preprocessing_file = st.session_state.get(f"{self.instance_id}_preprocessing_file", None)

        # State dict file
        state_dict_file = st.file_uploader(
            "Cargar el archivo de pesos del modelo de PyTorch", 
            accept_multiple_files=False, 
            type="pt", 
            key=f"coxtime_state_dict_file_uploader_{self.instance_id}"
        )
        if state_dict_file:
            st.session_state[f"{self.instance_id}_state_dict_file"] = state_dict_file
        self.state_dict_file = st.session_state.get(f"{self.instance_id}_state_dict_file", None)

        # Baseline hazards file
        baseline_hazards_file = st.file_uploader(
            "Cargar archivo de linea base de riesgos", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"coxtime_baseline_hazards_file_uploader_{self.instance_id}"
        )
        if baseline_hazards_file:
            st.session_state[f"{self.instance_id}_baseline_hazards_file"] = baseline_hazards_file
        self.baseline_hazards_file = st.session_state.get(f"{self.instance_id}_baseline_hazards_file", None)
    
        # State dict file
        labtrans_file = st.file_uploader(
            "Cargar el archivo con el modelo de traducción de etiquetas (labtrans)", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"coxtime_labtrans_file_uploader_{self.instance_id}"
        )
        if labtrans_file:
            st.session_state[f"{self.instance_id}_labtrans_file"] = labtrans_file
        self.labtrans_file = st.session_state.get(f"{self.instance_id}_labtrans_file", None)

        #Model Evaluation file
        evaluation_file = st.file_uploader(
            "Cargar archivo de evaluacion del modelo",
            accept_multiple_files=False,
            type="joblib",
            key=f"coxcc_evaluation_file_uploader_{self.instance_id}"
        )
        if evaluation_file:
            st.session_state[f"{self.instance_id}_evaluation_file"] = evaluation_file
        self.evaluation_file = st.session_state.get(f"{self.instance_id}_evaluation_file", None)
    
    def predict(self) -> pd.DataFrame:
        isReady = True
        if self.state_dict_file is None:
            st.warning("❗ Por favor, cargue el archivo de pesos del modelo para continuar")
            isReady = False
        if self.preprocessing_file is None:
            st.warning("❗ Por favor, cargue el archivo del modelo de preprocesamiento para continuar.")
            isReady = False
        if self.labtrans_file is None:
            st.warning("❗ Por favor, cargue el archivo del modelo de traducción de etiquetas (labtrans) para continuar.")
            isReady = False
        if self.baseline_hazards_file is None:
            st.warning("❗ Por favor, cargue el archivo de linea base de riesgos para continuar.")
            isReady = False
        # if self.net_file is None:
        #     st.warning("❗ Por favor, cargue el archivo de la red neuronal (net)para continuar.")
        #     isReady = False
        if self.evaluation_file is None:
            st.warning("❗ Por favor, cargue el archivo de evaluacion para continuar.")
            isReady = False
        if self.input_df is None:
            st.warning("❗ Por favor, cargue el archivo con el conjunto de datos de entrada.")
            isReady = False
        if not isReady:
            return pd.DataFrame()

        # self.net = torch.load(self.net_file, weights_only= False, map_location='cpu', pickle_module=pickle) #type: ignore
        self.state_dict = torch.load(self.state_dict_file, weights_only=True) #type: ignore
        self.net.load_state_dict(self.state_dict)
        self.loaded_labtrans = joblib.load(self.labtrans_file)
        self.loaded_preprocessor = joblib.load(self.preprocessing_file)
        self.loaded_baseline_hazards = joblib.load(self.baseline_hazards_file)


        self.loaded_model = CoxTime(self.net, optimizer=tt.optim.Adam, labtrans=self.loaded_labtrans) # type: ignore

        X_preprocessed = self.loaded_preprocessor.transform(self.input_df).astype('float32')
        self.loaded_model.baseline_hazards_ = self.loaded_baseline_hazards
        self.loaded_model.baseline_cumulative_hazards_ = self.loaded_model.compute_baseline_cumulative_hazards(
            set_hazards=True, 
            baseline_hazards_=self.loaded_baseline_hazards
        )
        times = self.loaded_baseline_hazards.index.values
        surv = self.loaded_model.predict_surv_df(X_preprocessed)
        surv.index = times
        self.surv = surv
        self.surv_curve = self._create_survival_curve()
        #creando objeto de evaluacion
        self.evaluation = joblib.load(self.evaluation_file)
        return surv

    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "Type": self.type,
            "Preprocessing File": self.preprocessing_file.name if self.preprocessing_file else "Not uploaded",
            "State Dict File": self.state_dict_file.name if self.state_dict_file else "Not uploaded",
            "Baseline Hazards File": self.baseline_hazards_file.name if self.baseline_hazards_file else "Not uploaded",
            "Net": self.net
        })
    
    def _create_survival_curve(self):
        baseline_survival = np.exp(-self.loaded_model.baseline_cumulative_hazards_) #type:ignore
        plt.figure(figsize=(10, 6))
        fig, ax = plt.subplots()
        # Línea dashed para la baseline
        ax.step(
            baseline_survival.index,
            baseline_survival.values,
            where='post',
            label='Baseline (población promedio)',
            color='black',
            linestyle='--',
            linewidth=2
        )
        # graficar cada curva de supervivencia
        for i in range(self.surv.shape[1]): #type:ignore
            ax.step(self.surv.index, self.surv.iloc[:, i],where='post',label=f'Empresa {i+1}') #type:ignore
        
        plt.ylabel('Probabilidad de supervivencia')
        plt.xlabel('Tiempo')
        plt.title('Curvas de supervivencia predichas (modelo CoxTime)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)
        return fig

    def get_survival_curve(self):
        return self.surv_curve
    
    def _median_times_simple(self, surv):
        times = surv.index.values.astype(float)   #type: ignore
        arr = surv.values   #type: ignore
        mts = {}
        
        for i in range(arr.shape[1]):
            idx = np.where(arr[:, i] <= 0.5)[0]
            median_time = float(times[idx[0]]) if idx.size else np.inf
            # Usar formato consistente "Caso X"
            mts[f"Caso {i+1}"] = median_time
        
        # Retornar como DataFrame
        return pd.DataFrame.from_dict(mts, orient='index', columns=['Median Survival Time']).astype(float)

    def predict_median_survival_time(self) -> DataFrame:
        if self.surv is None:
            st.warning("❗ Debe predecir primero")
            return pd.DataFrame()
        self.median_time = self._median_times_simple(self.surv)
        return self.median_time
    
    def concordance_index(self)-> float | None:
        if self.evaluation is None:
            if self.evaluation_file is None:
                st.warning("Por favor, cargue el archivo de evaluación para continuar.")
                return None
            else:
                self.evaluation = joblib.load(self.evaluation_file)

        self.c_index = self.evaluation.concordance_td()#type:ignore
        return self.c_index

    def get_survival_func(self):
        return self.surv
