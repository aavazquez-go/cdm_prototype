import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pandas import DataFrame
#from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt
import joblib

from pycox.datasets import metabric
from pycox.models import CoxPH, CoxCC
from pycox.evaluation import EvalSurv

from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self.name = ""
        self.type = ""
        self.parameters = {}
        self.input_df = None
    
    def set_name(self, name):
        self.name = name
    
    def set_type(self, model_type):
        self.type = model_type
    
    def set_parameters(self, parameters):
        self.parameters = parameters
    
    def set_input_df(self, df):
        self.input_df = df
    
    @abstractmethod
    def show_ui(self)->None:
        pass

    @abstractmethod
    def predict(self, dataframe) -> DataFrame:
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

class DeepSurvModel(Model):
    instance_count = 0
    def __init__(self, tab_index = None):
        super().__init__()
        DeepSurvModel.instance_count += 1
        self.type = "DeepSurv"
        self.tab_index = tab_index
        self.instance_id = f"deepsurv_tab_{tab_index}" if tab_index is not None else "deepsurv_temp"
        self.net  = tt.practical.MLPVanilla(in_features=45, num_nodes=[32, 32], out_features = 1, batch_norm=True, dropout=0.1, output_bias=False)
        self.preprocessing_file = None
        self.state_dict_file = None
        self.baseline_hazards_file = None
        self.surv_curve = None
        self.surv = None
        self.model = None
        self.model_loaded = None
        self.preprocessed_input = None
    
    def show_ui(self):
        st.markdown("#### DeepSurv")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Upload preprocessing model file", 
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
            "Upload PyTorch weight models", 
            accept_multiple_files=False, 
            type="pt", 
            key=f"deepsurv_state_dict_file_uploader_{self.instance_id}"
        )
        if state_dict_file:
            st.session_state[f"{self.instance_id}_state_dict_file"] = state_dict_file
        self.state_dict_file = st.session_state.get(f"{self.instance_id}_state_dict_file", None)

        # Baseline hazards file
        baseline_hazards_file = st.file_uploader(
            "Upload baseline hazards file", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"deepsurv_baseline_hazards_file_uploader_{self.instance_id}"
        )
        if baseline_hazards_file:
            st.session_state[f"{self.instance_id}_baseline_hazards_file"] = baseline_hazards_file
        self.baseline_hazards_file = st.session_state.get(f"{self.instance_id}_baseline_hazards_file", None)
    
    def predict(self, dataframe)->DataFrame:
        isReady = True
        if self.state_dict_file is None:
            st.warning("❗ Please upload PyTorch weight models to continue")
            isReady = False
        if self.preprocessing_file is None:
            st.warning("❗ Please upload preprocessing file to continue.")
            isReady = False
        if self.baseline_hazards_file is None:
            st.warning("❗ Please upload baseline hazards file to continue.")
            isReady = False
        if self.input_df is None:
            st.warning("❗ Please upload input dataset to continue.")
            isReady = False
        if not isReady:
            return pd.DataFrame()

        net = self.net
        state_dict = torch.load(self.state_dict_file, weights_only=True)
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
        self.preprocessed_input = pd.DataFrame(X, columns=input_df.columns)
        surv = model_loaded.predict_surv_df(X)
        self.surv = surv
        self.surv_curve = self._create_survival_curve(surv, model_loaded)
        # self.model = model_loaded
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
        
        return pd.DataFrame.from_dict(emp, orient='index', columns=['Median Survival Time'])
        
    def _predict_median_survival_time(self, surv):
        if self.surv is None:
            st.warning("❗ Please run prediction first to get median survival time.")
            return np.inf
        times = surv.index.values
        s = surv.iloc[:, 0].values
        idx = np.where(s <= 0.5)[0]
        return times[idx[0]] if len(idx) > 0 else np.inf

class LifelinesCoxPHModel(Model):
    
    def __init__(self, tab_index = None):
        super().__init__()
        self.type = "Lifelines-CoxPH"
        self.instance_id = f"lifelines_coxph_tab_{tab_index}" if tab_index is not None else "lifelines_coxph_temp"
        self.curv = None
        self.surv_curve = None
        self.model_file = None
        self.surv = None
        self.model_loaded = None
        self.preprocessing_file = None
        self.preprocessed_input = None
    
    def show_ui(self):
        st.markdown("### Lifelines CoxPHFitter")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Upload preprocessing model file", 
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
            "Upload CoxPHFitter trained model file",
            accept_multiple_files=False,
            type="joblib",
            key=f"lifelines_coxph_model_uploader_{self.instance_id}"
        )
        if model_file:
            st.session_state[f"{self.instance_id}_model_file"] = model_file
        elif st.session_state.get(f"{self.instance_id}_model_file", None):
            st.info(f"Archivo cargado: {st.session_state[f'{self.instance_id}_model_file'].name}")
        self.model_file = st.session_state.get(f"{self.instance_id}_model_file", None)
    
    def predict(self, dataframe) -> pd.DataFrame:
        if self.input_df is None:
            st.warning("❗ Please upload input dataset to continue.")
            return pd.DataFrame()
        if self.model_file is None:
            st.warning("❗ Please upload CoxPHFitter trained model file to continue.")
            return pd.DataFrame()
        if self.preprocessing_file is None:
            st.warning("❗ Please upload preprocessing file to continue.")
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
            st.warning("❗ Please run prediction first to get median survival time.")
            return pd.DataFrame()
        median_df = self.model_loaded.predict_median(self.preprocessed_input)
        return median_df

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
        self.surv_curve = None
        self.surv = None
        self.net = tt.practical.MLPVanilla(
            45,
            [32, 32],
            1,
            batch_norm = True,
            dropout = 0.1,
            output_bias=False
        )
        self.preprocessed_input = None
    
    def show_ui(self):
        st.markdown("### CoxCC ")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Upload preprocessing model file", 
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
            "Upload PyTorch weight models", 
            accept_multiple_files=False, 
            type="pt", 
            key=f"deepsurv_state_dict_file_uploader_{self.instance_id}"
        )
        if state_dict_file:
            st.session_state[f"{self.instance_id}_state_dict_file"] = state_dict_file
        self.state_dict_file = st.session_state.get(f"{self.instance_id}_state_dict_file", None)

        # Baseline hazards file
        baseline_hazards_file = st.file_uploader(
            "Upload baseline hazards file", 
            accept_multiple_files=False, 
            type="joblib", 
            key=f"deepsurv_baseline_hazards_file_uploader_{self.instance_id}"
        )
        if baseline_hazards_file:
            st.session_state[f"{self.instance_id}_baseline_hazards_file"] = baseline_hazards_file
        self.baseline_hazards_file = st.session_state.get(f"{self.instance_id}_baseline_hazards_file", None)

    def predict(self, dataframe) -> pd.DataFrame:
        isReady = True
        if self.state_dict_file is None:
            st.warning("❗ Please upload PyTorch weight models to continue")
            isReady = False
        if self.preprocessing_file is None:
            st.warning("❗ Please upload preprocessing file to continue.")
            isReady = False
        if self.baseline_hazards_file is None:
            st.warning("❗ Please upload baseline hazards file to continue.")
            isReady = False
        if self.input_df is None:
            st.warning("❗ Please upload input dataset to continue.")
            isReady = False
        if not isReady:
            return pd.DataFrame()
        
        loaded_preprocessor = joblib.load(self.preprocessing_file)
        loaded_baseline_hazards = joblib.load(self.baseline_hazards_file)
        loaded_model_net_state_dict = torch.load(self.state_dict_file)

        # print(f"Loaded preprocessor: {loaded_preprocessor}")
        # print(f"Loaded baseline hazards: {loaded_baseline_hazards}")
        # print(f"Loaded model state dict keys: {loaded_model_net_state_dict}")

        net = self.net
        net.load_state_dict(loaded_model_net_state_dict)
        model_loaded = CoxCC(net, tt.optim.Adam)
        
        # cargando el baseline hazards
        model_loaded.baseline_hazards_ = loaded_baseline_hazards
        model_loaded.baseline_cumulative_hazards_ = model_loaded.compute_baseline_cumulative_hazards(
            set_hazards=False, 
            baseline_hazards_= loaded_baseline_hazards
        )
        
        X_preprocessed = loaded_preprocessor.transform(self.input_df).astype('float32')
        # X_preprocessed_df = pd.DataFrame(X_preprocessed, columns = self.input_df.columns)
        self.preprocessed_input = pd.DataFrame(X_preprocessed, columns = self.input_df.columns)

        surv = model_loaded.predict_surv_df(X_preprocessed)
        print(f"Predicted survival values: {surv}")
        self.surv = surv
        self.surv_curve = self._create_survival_curve(surv, model_loaded)
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
        
        return pd.DataFrame.from_dict(emp, orient='index', columns=['Median Survival Time'])

    def _predict_median_survival_time(self, surv):
        if self.surv is None:
            st.warning("❗ Please run prediction first to get median survival time.")
            return np.inf
        times = surv.index.values
        s = surv.iloc[:, 0].values
        idx = np.where(s <= 0.5)[0]
        return times[idx[0]] if len(idx) > 0 else np.inf

class CoxTimeModel(Model):
    def __init__(self):
        super().__init__()
        self.type = "CoxTime"
        self.surv_curve = None
    
    def show_ui(self):
        st.markdown("### CoxTime")
    
    def predict(self, dataframe) -> pd.DataFrame:
        df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
        return df

    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "type": self.type,
        })
    def get_survival_curve(self):
        return self.surv_curve
    
    def predict_median_survival_time(self) -> DataFrame:
        df = pd.DataFrame(np.random.rand(5, 1), columns=['Median Survival Time'])
        return df


