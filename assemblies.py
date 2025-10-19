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
        
        # Convertir todos a mismo índice
        all_surv = pd.concat(survival_funcs.values(), axis=1)
        
        # Promediar por fila (por tiempo)
        ensemble_surv = all_surv.groupby(level=0).mean()
        
        # Calcular medianas del ensamble
        ensemble_median = self._calculate_medians(ensemble_surv)
        
        return ensemble_surv, ensemble_median
    
    def _calculate_medians(self, surv_df: pd.DataFrame) -> pd.Series:
        medians = {}
        for col in surv_df.columns:
            times = surv_df.index.values
            s = surv_df[col].values
            idx = np.where(s <= 0.5)[0]
            medians[col] = times[idx[0]] if len(idx) > 0 else np.inf
        return pd.Series(medians)
    
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
        
        # Concatenar todas las medianas
        all_medians = pd.concat(median_times.values(), axis=1)
        
        # Promediar por fila (por empresa/caso)
        ensemble_median = all_medians.mean(axis=1)
        
        # Para la curva de supervivencia, usar el promedio de las curvas disponibles
        if survival_funcs:
            all_surv = pd.concat(survival_funcs.values(), axis=1)
            ensemble_surv = all_surv.groupby(level=0).mean()
        else:
            ensemble_surv = pd.DataFrame()
        
        return ensemble_surv, ensemble_median
    
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
        
        for i, (model_name, surv) in enumerate(survival_funcs.items()):
            c_index = c_indices.get(model_name, 0.5)
            if c_index is None:
                c_index = 0.5
            self.weights[model_name] = c_index / total_c
        
        # Combinar funciones de supervivencia con pesos
        ensemble_surv = self._weighted_combine(survival_funcs, self.weights)
        
        # Combinar medianas con pesos
        all_medians = pd.concat(median_times.values(), axis=1)
        ensemble_median = (all_medians * pd.Series(list(self.weights.values()))).sum(axis=1)
        
        return ensemble_surv, ensemble_median
    
    def _weighted_combine(self, survival_funcs: Dict, weights: Dict) -> pd.DataFrame:
        weighted_surv = None
        for model_name, weight in weights.items():
            surv = survival_funcs[model_name]
            if weighted_surv is None:
                weighted_surv = surv * weight
            else:
                # Alinear índices si es necesario
                weighted_surv = weighted_surv.add(surv * weight, fill_value=0)
        return weighted_surv
    
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
        
        # Obtener rangos (mayor c_index = rank menor = peso mayor)
        sorted_models = sorted(c_values.items(), key=lambda x: x[1], reverse=True)
        n_models = len(sorted_models)
        
        # Asignar pesos inversamente proporcionales al rango
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            self.weights[model_name] = (n_models - rank + 1) / sum(range(1, n_models + 1))
        
        # Combinar con pesos
        weighted_surv = self._weighted_combine(survival_funcs, self.weights)
        
        all_medians = pd.concat(median_times.values(), axis=1)
        ensemble_median = (all_medians * pd.Series(list(self.weights.values()))).sum(axis=1)
        
        return weighted_surv, ensemble_median
    
    def _weighted_combine(self, survival_funcs: Dict, weights: Dict) -> pd.DataFrame:
        weighted_surv = None
        for model_name, weight in weights.items():
            surv = survival_funcs[model_name]
            if weighted_surv is None:
                weighted_surv = surv * weight
            else:
                weighted_surv = weighted_surv.add(surv * weight, fill_value=0)
        return weighted_surv
    
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
        all_medians = pd.concat(median_times.values(), axis=1)
        
        # Para simplificar, usamos pesos basados en c_index como meta-modelo
        # En una implementación más avanzada, se entrenaría un modelo real
        c_values = {k: v if v is not None else 0.5 for k, v in c_indices.items()}
        c_array = np.array([c_values[k] for k in median_times.keys()])
        c_normalized = c_array / np.sum(c_array)
        
        # Combinar medianas
        ensemble_median = (all_medians * c_normalized).sum(axis=1)
        
        # Combinar supervivencia
        all_surv = pd.concat(survival_funcs.values(), axis=1)
        ensemble_surv = (all_surv.groupby(level=0).apply(
            lambda x: (x * c_normalized).sum(axis=1)
        ))
        
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
        
        all_medians = pd.concat(median_times.values(), axis=1)
        ensemble_median = (all_medians * pd.Series(list(self.weights.values()))).sum(axis=1)
        
        return weighted_surv, ensemble_median
    
    def _weighted_combine(self, survival_funcs: Dict, weights: Dict) -> pd.DataFrame:
        weighted_surv = None
        for model_name, weight in weights.items():
            surv = survival_funcs[model_name]
            if weighted_surv is None:
                weighted_surv = surv * weight
            else:
                weighted_surv = weighted_surv.add(surv * weight, fill_value=0)
        return weighted_surv
    
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
        
        # Ranquear empresas por mediana de supervivencia de cada modelo
        all_medians = pd.concat(median_times.values(), axis=1)
        
        # Obtener ranks para cada modelo
        ranks = all_medians.rank(axis=0)
        
        # Promedio de ranks (consensus ranking)
        consensus_rank = ranks.mean(axis=1)
        
        # Usar el rank de consenso para ordenar
        # Mapear de vuelta a tiempo (usar promedio de medianas)
        ensemble_median = all_medians.mean(axis=1)
        
        # Para la curva de supervivencia
        all_surv = pd.concat(survival_funcs.values(), axis=1)
        ensemble_surv = all_surv.groupby(level=0).mean()
        
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
            
            print(f"Survival function for {model_key}: {model.surv}")


            # Obtener función de supervivencia
            if hasattr(model, 'surv') and model.surv is not None:
                self.survival_funcs[model_key] = model.surv
            
            if hasattr(model,'median_time') and model.median_time is not None:
                try:
                    median_df = model.median_time
                    if not median_df.empty:
                        self.median_times[model_key] = median_df.iloc[:, 0]
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