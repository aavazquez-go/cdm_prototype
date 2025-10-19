import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from abc import ABC, abstractmethod
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple
from settings import Model

class EnsembleStrategy(ABC):
    """Clase base abstracta para estrategias de ensamble"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Combina predicciones de múltiples modelos
        
        Args:
            models: Lista de modelos entrenados
            survival_funcs: Dict con DataFrames de curvas de supervivencia por modelo
            median_times: Dict con medianas de tiempo de supervivencia por modelo
            c_indices: Dict con índices de concordancia por modelo
            
        Returns:
            Tuple con (DataFrame de supervivencia combinada, Series de medianas combinadas)
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """Retorna los pesos utilizados por el modelo"""
        pass

class SurvivalFunctionAveraging(EnsembleStrategy):
    """Promedia las funciones de supervivencia de todos los modelos"""
    
    def __init__(self):
        super().__init__("Survival Function Averaging")
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if not survival_funcs:
            st.error("No hay funciones de supervivencia disponibles")
            return pd.DataFrame(), pd.Series()
        
        # Alinear todos los índices de tiempo
        all_times = []
        for surv in survival_funcs.values():
            all_times.extend(surv.index.tolist())
        all_times = sorted(set(all_times))
        
        # Reindexar todos los DataFrames al índice común
        aligned_surv = []
        for surv in survival_funcs.values():
            aligned = surv.reindex(all_times, method='ffill')
            aligned_surv.append(aligned)
        
        # Concatenar por columnas
        all_surv_concat = pd.concat(aligned_surv, axis=1)
        
        # Agrupar por empresa (última parte del nombre de columna después del último _)
        # y promediar
        ensemble_surv = all_surv_concat.groupby( #type: ignore
            lambda x: x.rsplit('_', 1)[-1], axis=1
        ).mean()
        
        # Calcular medianas a partir de la curva promediada
        ensemble_median = self._calculate_medians(ensemble_surv)
        
        return ensemble_surv, ensemble_median
    
    def _calculate_medians(self, surv_df: pd.DataFrame) -> pd.Series:
        """Calcula los tiempos medianos de supervivencia de cada empresa"""
        medians = {}
        for col in surv_df.columns:
            times = surv_df.index.values
            s = surv_df[col].values
            
            # Buscar el primer tiempo donde la supervivencia cae por debajo de 0.5
            idx = np.where(s <= 0.5)[0]
            
            if len(idx) > 0:
                medians[col] = times[idx[0]]
            else:
                # Si nunca cae por debajo de 0.5, usar infinity
                medians[col] = np.inf
        
        return pd.Series(medians, name='Median Survival Time')
    
    def get_weights(self) -> Dict[str, float]:
        return {"method": "Equal weights for all models"}

class MedianTimeAveraging(EnsembleStrategy):
    """Promedia los tiempos de supervivencia mediana de todos los modelos"""
    
    def __init__(self):
        super().__init__("Median Time Averaging")
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if not median_times:
            st.error("No hay tiempos medianos disponibles")
            return pd.DataFrame(), pd.Series()
        
        # Combinar medianas de manera correcta
        ensemble_median = self._average_median_combine(median_times)
        
        # Para la curva de supervivencia, usar el promedio de las curvas disponibles
        if survival_funcs:
            # Alinear índices
            all_times = []
            for surv in survival_funcs.values():
                all_times.extend(surv.index.tolist())
            all_times = sorted(set(all_times))
            
            aligned_surv = []
            for surv in survival_funcs.values():
                aligned = surv.reindex(all_times, method='ffill')
                aligned_surv.append(aligned)
            
            all_surv_concat = pd.concat(aligned_surv, axis=1)
            ensemble_surv = all_surv_concat.groupby(
                lambda x: x.rsplit('_', 1)[-1], axis=1
            ).mean()
        else:
            ensemble_surv = pd.DataFrame()
        
        return ensemble_surv, ensemble_median
    
    def _average_median_combine(self, median_times: Dict) -> pd.Series:
        """Promedia las medianas de manera correcta"""
        if not median_times:
            return pd.Series(name='Median Survival Time')
        
        # Extraer el nombre del caso de cada índice y agrupar antes de concatenar
        case_medians = {}
        
        for model_name, median_series in median_times.items():
            for idx, value in median_series.items():
                # idx tiene formato "ModelName_Caso X"
                # Extraer solo la parte "Caso X"
                case_name = idx.rsplit('_', 1)[-1]
                
                if case_name not in case_medians:
                    case_medians[case_name] = []
                
                case_medians[case_name].append(value)
        
        # Promediar los valores para cada caso
        ensemble_median = {}
        for case_name, values in case_medians.items():
            ensemble_median[case_name] = np.mean(values)
        
        result = pd.Series(ensemble_median, name='Median Survival Time')
        return result
    
    def get_weights(self) -> Dict[str, float]:
        return {"method": "Equal weights for median averaging"}

class PerformanceWeightedEnsemble(EnsembleStrategy):
    """Pondera las predicciones basándose en el índice de concordancia de cada modelo"""
    
    def __init__(self):
        super().__init__("Performance-weighted ensemble")
        self.weights = {}
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if not survival_funcs or not c_indices:
            st.error("Se requieren funciones de supervivencia e índices de concordancia")
            return pd.DataFrame(), pd.Series()
        
        # Calcular pesos basados en c_index (normalizar)
        c_values = np.array([v for v in c_indices.values() if v is not None])
        if len(c_values) == 0 or np.sum(c_values) <= 0:
            st.warning("No se pueden calcular pesos válidos. Usando pesos iguales.")
            return SurvivalFunctionAveraging().combine_predictions(
                models, survival_funcs, median_times, c_indices)
        
        self.weights = {}
        total_c = np.sum(c_values)
        
        for model_name, surv in survival_funcs.items():
            c_index = c_indices.get(model_name, 0.5)
            if c_index is None:
                c_index = 0.5
            self.weights[model_name] = c_index / total_c
        
        # Combinar funciones de supervivencia con pesos
        ensemble_surv = self._weighted_combine(survival_funcs, self.weights)
        
        # Combinar medianas con pesos - método mejorado
        ensemble_median = self._weighted_median_combine(median_times, self.weights)
        
        return ensemble_surv, ensemble_median
    
    def _weighted_combine(self, survival_funcs: Dict, weights: Dict) -> pd.DataFrame:
        # Alinear índices de tiempo
        all_times = []
        for surv in survival_funcs.values():
            all_times.extend(surv.index.tolist())
        all_times = sorted(set(all_times))
        
        weighted_surv = None
        for model_name, weight in weights.items():
            surv = survival_funcs[model_name]
            aligned = surv.reindex(all_times, method='ffill')
            weighted = aligned * weight
            
            if weighted_surv is None:
                weighted_surv = weighted
            else:
                weighted_surv = weighted_surv.add(weighted, fill_value=0)
        
        # Agrupar por empresa
        ensemble_surv = weighted_surv.groupby(
            lambda x: x.rsplit('_', 1)[-1], axis=1
        ).sum()
        
        return ensemble_surv
    
    def _weighted_median_combine(self, median_times: Dict, weights: Dict) -> pd.Series:
        """Combina medianas ponderadas de manera correcta"""
        if not median_times:
            return pd.Series()
        
        # Crear un DataFrame temporal con todas las medianas
        all_medians_list = []
        for model_name, median_series in median_times.items():
            # median_series tiene índice tipo "DeepSurv_Caso 1", "DeepSurv_Caso 2", etc.
            weight = weights.get(model_name, 1.0)
            weighted_median = median_series * weight
            all_medians_list.append(weighted_median)
        
        # Concatenar todas las medianas ponderadas
        all_medians_concat = pd.concat(all_medians_list)
        
        # Agrupar por el nombre del caso (última parte después del último _)
        # y sumar (ya que están ponderadas)
        ensemble_median = all_medians_concat.groupby(
            lambda x: x.rsplit('_', 1)[-1]
        ).sum()
        
        ensemble_median.name = 'Median Survival Time'
        return ensemble_median
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights

class RankBasedEnsemble(EnsembleStrategy):
    """Ranquea los modelos y combina basándose en su desempeño relativo"""
    
    def __init__(self):
        super().__init__("Rank-based ensemble")
        self.weights = {}
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if not survival_funcs or not c_indices:
            st.error("Se requieren funciones de supervivencia e índices de concordancia")
            return pd.DataFrame(), pd.Series()
        
        # Ranquear modelos por c_index
        c_values = {k: v if v is not None else 0.5 for k, v in c_indices.items()}
        
        sorted_models = sorted(c_values.items(), key=lambda x: x[1], reverse=True)
        n_models = len(sorted_models)
        
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            self.weights[model_name] = (n_models - rank + 1) / sum(range(1, n_models + 1))
        
        # Combinar con pesos
        weighted_surv = self._weighted_combine(survival_funcs, self.weights)
        
        # Combinar medianas con pesos - método mejorado
        ensemble_median = self._weighted_median_combine(median_times, self.weights)
        
        return weighted_surv, ensemble_median
    
    def _weighted_combine(self, survival_funcs: Dict, weights: Dict) -> pd.DataFrame:
        all_times = []
        for surv in survival_funcs.values():
            all_times.extend(surv.index.tolist())
        all_times = sorted(set(all_times))
        
        weighted_surv = None
        for model_name, weight in weights.items():
            surv = survival_funcs[model_name]
            aligned = surv.reindex(all_times, method='ffill')
            weighted = aligned * weight
            
            if weighted_surv is None:
                weighted_surv = weighted
            else:
                weighted_surv = weighted_surv.add(weighted, fill_value=0)
        
        ensemble_surv = weighted_surv.groupby(
            lambda x: x.rsplit('_', 1)[-1], axis=1
        ).sum()
        
        return ensemble_surv
    
    def _weighted_median_combine(self, median_times: Dict, weights: Dict) -> pd.Series:
        """Combina medianas ponderadas de manera correcta"""
        if not median_times:
            return pd.Series()
        
        # Crear un DataFrame temporal con todas las medianas
        all_medians_list = []
        for model_name, median_series in median_times.items():
            weight = weights.get(model_name, 1.0)
            weighted_median = median_series * weight
            all_medians_list.append(weighted_median)
        
        # Concatenar todas las medianas ponderadas
        all_medians_concat = pd.concat(all_medians_list)
        
        # Agrupar por el nombre del caso (última parte después del último _)
        # y sumar (ya que están ponderadas)
        ensemble_median = all_medians_concat.groupby(
            lambda x: x.rsplit('_', 1)[-1]
        ).sum()
        
        ensemble_median.name = 'Median Survival Time'
        return ensemble_median
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights

class StackingEnsemble(EnsembleStrategy):
    """Usa las predicciones como características para entrenar un meta-modelo"""
    
    def __init__(self):
        super().__init__("Stacking")
        self.meta_model = None
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if len(survival_funcs) < 2:
            st.warning("Se requieren al menos 2 modelos para stacking")
            return SurvivalFunctionAveraging().combine_predictions(
                models, survival_funcs, median_times, c_indices)
        
        # Usar medianas como características
        all_medians = pd.concat(median_times.values(), axis=0)
        
        # Ponderar por c_index
        c_values = {k: v if v is not None else 0.5 for k, v in c_indices.items()}
        c_array = np.array([c_values[k] for k in median_times.keys()])
        c_normalized = c_array / np.sum(c_array)
        
        ensemble_median = (all_medians * c_normalized[0]).groupby(
            lambda x: x.rsplit('_', 1)[-1]
        ).sum()
        
        # Combinar supervivencia
        all_times = []
        for surv in survival_funcs.values():
            all_times.extend(surv.index.tolist())
        all_times = sorted(set(all_times))
        
        aligned_surv = []
        for surv in survival_funcs.values():
            aligned = surv.reindex(all_times, method='ffill')
            aligned_surv.append(aligned)
        
        all_surv_concat = pd.concat(aligned_surv, axis=1)
        ensemble_surv = all_surv_concat.groupby(
            lambda x: x.rsplit('_', 1)[-1], axis=1
        ).mean()
        
        return ensemble_surv, ensemble_median
    
    def get_weights(self) -> Dict[str, float]:
        return {"method": "Meta-learner weights based on c_index"}

class BayesianModelAveraging(EnsembleStrategy):
    """Promedia bayesianamente ponderando por el desempeño de cada modelo"""
    
    def __init__(self):
        super().__init__("Bayesian Model Averaging")
        self.weights = {}
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if not survival_funcs or not c_indices:
            st.error("Se requieren funciones de supervivencia e índices de concordancia")
            return pd.DataFrame(), pd.Series()
        
        # Calcular pesos con transformación suave de c_index
        c_values = {k: v if v is not None else 0.5 for k, v in c_indices.items()}
        
        # Usar exponencial para acentuar diferencias
        exp_c = np.exp([c_values[k] * 5 for k in c_values.keys()])
        total_exp_c = np.sum(exp_c)
        
        for i, model_name in enumerate(c_values.keys()):
            self.weights[model_name] = exp_c[i] / total_exp_c
        
        # Combinar con pesos
        weighted_surv = self._weighted_combine(survival_funcs, self.weights)
        
        # Combinar medianas con pesos - método mejorado
        ensemble_median = self._weighted_median_combine(median_times, self.weights)
        
        return weighted_surv, ensemble_median
    
    def _weighted_combine(self, survival_funcs: Dict, weights: Dict) -> pd.DataFrame:
        all_times = []
        for surv in survival_funcs.values():
            all_times.extend(surv.index.tolist())
        all_times = sorted(set(all_times))
        
        weighted_surv = None
        for model_name, weight in weights.items():
            surv = survival_funcs[model_name]
            aligned = surv.reindex(all_times, method='ffill')
            weighted = aligned * weight
            
            if weighted_surv is None:
                weighted_surv = weighted
            else:
                weighted_surv = weighted_surv.add(weighted, fill_value=0)
        
        ensemble_surv = weighted_surv.groupby(
            lambda x: x.rsplit('_', 1)[-1], axis=1
        ).sum()
        
        return ensemble_surv
    
    def _weighted_median_combine(self, median_times: Dict, weights: Dict) -> pd.Series:
        """Combina medianas ponderadas de manera correcta"""
        if not median_times:
            return pd.Series()
        
        # Crear un DataFrame temporal con todas las medianas
        all_medians_list = []
        for model_name, median_series in median_times.items():
            weight = weights.get(model_name, 1.0)
            weighted_median = median_series * weight
            all_medians_list.append(weighted_median)
        
        # Concatenar todas las medianas ponderadas
        all_medians_concat = pd.concat(all_medians_list)
        
        # Agrupar por el nombre del caso (última parte después del último _)
        # y sumar (ya que están ponderadas)
        ensemble_median = all_medians_concat.groupby(
            lambda x: x.rsplit('_', 1)[-1]
        ).sum()
        
        ensemble_median.name = 'Median Survival Time'
        return ensemble_median
    
    def get_weights(self) -> Dict[str, float]:
        return self.weights

class VotingRankingConsensus(EnsembleStrategy):
    """Combina votos de modelos basándose en ranking de supervivencia"""
    
    def __init__(self):
        super().__init__("Voting/Ranking Consensus")
    
    def combine_predictions(self, models: List[Model], survival_funcs: Dict, 
                           median_times: Dict, c_indices: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        if not median_times:
            st.error("No hay medianas disponibles para ranking")
            return pd.DataFrame(), pd.Series()
        
        all_medians = pd.concat(median_times.values(), axis=0)
        ensemble_median = all_medians.groupby(
            lambda x: x.rsplit('_', 1)[-1]
        ).mean()
        
        # Para la curva de supervivencia
        if survival_funcs:
            all_times = []
            for surv in survival_funcs.values():
                all_times.extend(surv.index.tolist())
            all_times = sorted(set(all_times))
            
            aligned_surv = []
            for surv in survival_funcs.values():
                aligned = surv.reindex(all_times, method='ffill')
                aligned_surv.append(aligned)
            
            all_surv_concat = pd.concat(aligned_surv, axis=1)
            ensemble_surv = all_surv_concat.groupby( #type:ignore
                lambda x: x.rsplit('_', 1)[-1], axis=1
            ).mean()
        else:
            ensemble_surv = pd.DataFrame()
        
        return ensemble_surv, ensemble_median
    
    def get_weights(self) -> Dict[str, float]:
        return {"method": "Consensus ranking from all models"}


class EnsembleManager:
    """Gestor central de ensambles"""
    
    STRATEGIES = {
        "Survival Function Averaging": SurvivalFunctionAveraging,
        "Median Time Averaging": MedianTimeAveraging,
        "Stacking": StackingEnsemble,
        "Performance-weighted ensemble": PerformanceWeightedEnsemble,
        "Rank-based ensemble": RankBasedEnsemble,
        "Bayesian Model Averaging": BayesianModelAveraging,
        "Voting/Ranking Consensus": VotingRankingConsensus,
    }
    
    def __init__(self, strategy_name: str):
        if strategy_name not in self.STRATEGIES:
            raise ValueError(f"Estrategia no soportada: {strategy_name}")
        self.strategy = self.STRATEGIES[strategy_name]()
        self.ensemble_surv = None
        self.ensemble_median = None
    
    def fit(self, models: List[Model]) -> None:
        """Extrae predicciones de modelos"""
        self.survival_funcs = {}
        self.median_times = {}
        self.c_indices = {}
        
        print(f"Models: {models}")

        for i, model in enumerate(models):
            if model is None:
                continue
            
            model_key = f"{model.name}_{i}"
            
            print(f"Survival function for {model_key}:\n {model.surv}")

            # Obtener función de supervivencia
            if hasattr(model, 'surv') and model.surv is not None:
                # Renombrar columnas para incluir el nombre del modelo
                surv_renamed = model.surv.copy()
                surv_renamed.columns = [f"{model.name}_{col}" for col in surv_renamed.columns]
                self.survival_funcs[model_key] = surv_renamed
            
            # Obtener median time
            if hasattr(model, 'median_time') and model.median_time is not None:
                try:
                    median_df = model.median_time
                    if not median_df.empty:
                        # Renombrar índice para incluir el nombre del modelo
                        median_series = median_df.iloc[:, 0].copy()
                        median_series.index = [f"{model.name}_{idx}" for idx in median_series.index]
                        self.median_times[model_key] = median_series
                except Exception as e:
                    st.warning(f"Error obteniendo la mediana de {model.name}: {e}")

            # Obtener c_index
            if hasattr(model, 'c_index'):
                try:
                    c_idx = model.c_index
                    self.c_indices[model_key] = c_idx
                except Exception as e:
                    st.warning(f"Error obteniendo c_index de {model.name}: {e}")
    
    def predict(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Genera predicción del ensamble"""
        self.ensemble_surv, self.ensemble_median = self.strategy.combine_predictions(
            [], self.survival_funcs, self.median_times, self.c_indices
        )
        
        # Limpiar nombres de columnas del resultado
        if not self.ensemble_surv.empty:
            # Extraer solo el índice de empresa (remover nombre del modelo)
            new_cols = []
            seen = {}
            for col in self.ensemble_surv.columns:
                # Si la columna tiene formato "nombre_modelo_0", extraer solo el número
                parts = col.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    base_col = parts[1]
                    if base_col not in seen:
                        seen[base_col] = 0
                    seen[base_col] += 1
                    new_cols.append(f"Empresa_{parts[1]}")
                else:
                    new_cols.append(col)
            
            self.ensemble_surv.columns = new_cols
        
        return self.ensemble_surv, self.ensemble_median
    
    def plot_ensemble_survival(self) -> plt.Figure:
        """Grafica las curvas de supervivencia del ensamble"""
        if self.ensemble_surv is None:
            st.error("Debe ejecutar predict() primero")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(self.ensemble_surv.shape[1]):
            ax.step(
                self.ensemble_surv.index,
                self.ensemble_surv.iloc[:, i],
                where='post',
                label=f'Empresa {i+1}',
                linewidth=2
            )
        
        ax.set_ylabel('Probabilidad de supervivencia')
        ax.set_xlabel('Tiempo')
        ax.set_title(f'Curvas de supervivencia del ensamble ({self.strategy.name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        return fig
    
    def get_strategy_info(self) -> Dict:
        """Retorna información del ensamble"""
        return {
            "strategy": self.strategy.name,
            "weights": self.strategy.get_weights(),
            "num_models": len(self.survival_funcs)
        }