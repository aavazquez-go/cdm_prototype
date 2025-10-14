import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pandas import DataFrame
#from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt
import joblib

from pycox.datasets import metabric
from pycox.models import CoxPH
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

class DeepSurvModel(Model):
    instance_count = 0
    def __init__(self):
        super().__init__()
        DeepSurvModel.instance_count += 1
        self.instance_id = f"deepsurv_{DeepSurvModel.instance_count}"
        self.type = "DeepSurv"
        self.net  = tt.practical.MLPVanilla(in_features=45, num_nodes=[32, 32], out_features = 1, batch_norm=True, dropout=0.1, output_bias=False)
        self.preprocessing_file = None
        self.state_dict_file = None
        self.baseline_hazards_file = None
        if self.instance_id not in st.session_state:
            st.session_state[self.instance_id] = self
    
    def show_ui(self):
        st.markdown("#### DeepSurv Model Parameters")
        # Preprocessing file
        preprocessing_file = st.file_uploader(
            "Upload preprocessing model file", 
            accept_multiple_files=False, 
            type="pkl", 
            key=f"deepsurv_preprocessing_file_uploader_{self.instance_id}"
        )
        if preprocessing_file:
            st.session_state[f"{self.instance_id}_preprocessing_file"] = preprocessing_file
            st.success("✅ Preprocessing file uploaded successfully.")
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
            st.success("✅ PyTorch weight model uploaded successfully.")
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
            st.success("✅ Baseline hazards file uploaded successfully.")
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
        input_df = self.input_df
        preprocessor = joblib.load(self.preprocessing_file)
        X = preprocessor.transform(input_df).astype('float32')
        surv = model_loaded.predict_surv_df(X)
        return surv
    
    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "Type": self.type,
            # "Net": self.net
            "Preprocessing File": self.preprocessing_file.name if self.preprocessing_file else "Not uploaded",
        })

class LifelinesCoxPHModel(Model):
    def __init__(self):
        super().__init__()
        self.type = "Lifelines-PHCox"
    
    def show_ui(self):
        st.markdown("### Lifelines CoxPHFitter")
    
    def predict(self, dataframe) -> pd.DataFrame:
        df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
        return df
    
    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "type": self.type,
        })

class CoxCCModel(Model):
    def __init__(self):
        super().__init__()
        self.type = "CoxCC"
    
    def show_ui(self):
        st.markdown("### CoxCC")
    
    def predict(self, dataframe) -> pd.DataFrame:
        df = pd.DataFrame(np.random.rand(5, 3), columns=['A', 'B', 'C'])
        return df
    
    def model_details(self) -> None:
        st.json({
            "Name": self.name,
            "type": self.type,
        })

class CoxTimeModel(Model):
    def __init__(self):
        super().__init__()
        self.type = "CoxTime"
    
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


