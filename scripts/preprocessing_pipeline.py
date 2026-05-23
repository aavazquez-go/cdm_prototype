"""
Pipeline de preprocesamiento para datos crudos de insolvencia.

Transforma datos en formato raw (con variables categóricas y missing values)
al formato de 60 columnas numéricas listo para modelos de supervivencia.

Uso:
    from scripts.preprocessing_pipeline import preprocess_raw_data
    df_proc = preprocess_raw_data(df_raw)

Artefactos cargados desde models/preprocessing/.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PREPROC_DIR = Path("models/preprocessing/")

_artifacts = None


def _load_artifacts():
    global _artifacts
    if _artifacts is not None:
        return _artifacts

    arts = {
        "mice_imputer": joblib.load(PREPROC_DIR / "mice_imputer.joblib"),
        "ohe": joblib.load(PREPROC_DIR / "one_hot_encoder.joblib"),
        "yeo_johnson": joblib.load(PREPROC_DIR / "yeo_johnson_transformer.joblib"),
        "ohe_columns": joblib.load(PREPROC_DIR / "ohe_columns.joblib"),
        "numerical_columns": joblib.load(PREPROC_DIR / "numerical_columns.joblib"),
        "expected_columns": joblib.load(PREPROC_DIR / "expected_columns.joblib"),
    }

    with open(PREPROC_DIR / "pipeline_metadata.json") as f:
        meta = json.load(f)

    arts["cat_columns"] = meta["categorical_columns"]
    arts["removed_cols"] = list(meta["initial_removed_columns"]) + ["Type"]

    _artifacts = arts
    return arts


FEATURE_EXCLUDE = {"CIF", "Start", "Stop", "Event"}


def preprocess_raw_data(df_raw):
    """
    Transforma datos crudos (mismo schema que raw_data.csv) al formato
    de features listo para modelos de supervivencia.

    Pasos:
        1. Elimina columnas no usadas (Unnamed:0, F26, F32, Type)
        2. Aplica MICE imputation a columnas numéricas
        3. Aplica One-Hot Encoding a columnas categóricas
        4. Aplica Yeo-Johnson normalization a columnas numéricas
        5. Ensambla DataFrame de 60 columnas en el orden de expected_columns

    Parameters
    ----------
    df_raw : pd.DataFrame
        Datos crudos con columnas del tipo N3 (str), N4 (str), N*, F*, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame con 60 columnas de features (excluye CIF, Start, Stop, Event)
        listo para aplicar el StandardScaler específico de cada modelo.
    """
    arts = _load_artifacts()
    df = df_raw.copy()

    # 1. Remover columnas no deseadas
    drop = [c for c in arts["removed_cols"] if c in df.columns]
    if drop:
        df.drop(columns=drop, inplace=True, errors="ignore")

    # 2. Preparar columnas numéricas (rellenar con 0 si faltan)
    num_cols = arts["numerical_columns"]
    df_num = pd.DataFrame(index=df.index)
    for col in num_cols:
        df_num[col] = df[col].astype(float) if col in df.columns else 0.0

    # 3. Preparar columnas categóricas (rellenar con '' si faltan)
    cat_cols = arts["cat_columns"]
    df_cat = pd.DataFrame(index=df.index)
    for col in cat_cols:
        df_cat[col] = df[col].astype(str) if col in df.columns else ""

    # 4. MICE imputation (solo numéricas, no refit)
    df_num_imputed = pd.DataFrame(
        arts["mice_imputer"].transform(df_num.values),
        columns=num_cols,
        index=df.index,
    )

    # 5. One-Hot Encoding (categóricas)
    df_cat_ohe = pd.DataFrame(
        arts["ohe"].transform(df_cat.values),
        columns=arts["ohe_columns"],
        index=df.index,
    ).astype(np.int64)

    # 6. Yeo-Johnson normalization (numéricas imputadas)
    df_num_norm = pd.DataFrame(
        arts["yeo_johnson"].transform(df_num_imputed.values),
        columns=num_cols,
        index=df.index,
    )

    # 7. Ensamblar en orden de expected_columns (solo features, sin CIF/Start/Stop/Event)
    feature_cols = [c for c in arts["expected_columns"] if c not in FEATURE_EXCLUDE]

    df_out = pd.DataFrame(0.0, index=df.index, columns=feature_cols)

    # Rellenar numéricas
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = df_num_norm[col].values

    # Rellenar OHE
    for col in arts["ohe_columns"]:
        if col in df_out.columns:
            df_out[col] = df_cat_ohe[col].values

    return df_out


def validate_columns(df):
    """Verifica que el DataFrame tenga las 60 columnas de features esperadas."""
    arts = _load_artifacts()
    feature_cols = [c for c in arts["expected_columns"] if c not in FEATURE_EXCLUDE]
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan {len(missing)} columnas en el DataFrame preprocesado: {sorted(missing)}")
    extra = set(df.columns) - set(feature_cols)
    if extra:
        raise ValueError(f"Sobran {len(extra)} columnas no esperadas: {sorted(extra)}")
    return True
