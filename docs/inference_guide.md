# Guía de Inferencia — Modelos de Supervivencia

## Requisitos

Activar el entorno virtual:

```bash
source /media/datos/WORKSPACE/DOCTORADO/cdm_prototype/.venv/bin/activate
```

---

## Pipeline de Preprocesamiento

La inferencia tiene **dos etapas de preprocesamiento**:

1. **Pipeline general** (MICE + OHE + Yeo-Johnson) — usando artefactos de `models/preprocessing/`
2. **StandardScaler específico por modelo** — cargado desde `notebooks/<modelo>_models/`

### Etapa 1 — Pipeline general (datos crudos → features normalizadas + one-hot)

Esta etapa toma datos crudos con el mismo formato que `raw_data.csv` y aplica las transformaciones aprendidas durante el entrenamiento.

```python
import joblib
import pandas as pd
import numpy as np

# ── Cargar artefactos del pipeline ──
PREPROC_DIR = "models/preprocessing/"

mice_imputer    = joblib.load(f"{PREPROC_DIR}mice_imputer.joblib")
ohe             = joblib.load(f"{PREPROC_DIR}one_hot_encoder.joblib")
yeo_johnson_pt  = joblib.load(f"{PREPROC_DIR}yeo_johnson_transformer.joblib")
ohe_columns     = joblib.load(f"{PREPROC_DIR}ohe_columns.joblib")
numerical_cols  = joblib.load(f"{PREPROC_DIR}numerical_columns.joblib")
expected_cols   = joblib.load(f"{PREPROC_DIR}expected_columns.joblib")
cat_columns     = ["N3", "N4", "N5", "N8", "N9", "N10", "N11", "N15"]
removed_cols    = ["Unnamed: 0", "F26", "F32", "Type"]

def preprocess_raw_data(df_raw):
    """
    Transforma datos crudos (mismo schema que raw_data.csv)
    al formato de 64 columnas listo para los modelos de supervivencia.
    Retorna (X_processed, df_full) donde X_processed es el DataFrame
    con las 64 columnas listo para el siguiente paso.
    """
    df = df_raw.copy()

    # 1. Eliminar columnas no usadas
    cols_to_drop = [c for c in removed_cols if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 2. Separar numéricas y categóricas
    cat_present = [c for c in cat_columns if c in df.columns]
    num_present = [c for c in numerical_cols if c in df.columns]

    df_num = df[num_present]
    df_cat = df[cat_present]

    # 3. MICE imputation (solo numéricas)
    df_num_imputed = pd.DataFrame(
        mice_imputer.transform(df_num),
        columns=num_present,
        index=df.index
    )

    # 4. One-Hot Encoding (categóricas)
    df_cat_ohe = pd.DataFrame(
        ohe.transform(df_cat),
        columns=ohe_columns,
        index=df.index
    )
    df_cat_ohe = df_cat_ohe.astype(np.int64)

    # 5. Yeo-Johnson normalization (numéricas)
    df_num_norm = pd.DataFrame(
        yeo_johnson_pt.transform(df_num_imputed),
        columns=num_present,
        index=df.index
    )

    # 6. Combinar y reordenar según expected_columns
    df_full = pd.concat([df_num_norm, df_cat_ohe], axis=1)
    # Asegurar que todas las columnas esperadas existan (rellenar con 0 si falta)
    for col in expected_cols:
        if col not in df_full.columns:
            df_full[col] = 0
    df_full = df_full[expected_cols]

    return df_full
```

> **Nota:** Si los datos de entrada contienen columnas de destino (`FCA`, `Y`, `Start`, `Stop`, `Event`), se conservan automáticamente durante el pipeline (no están en `removed_cols`). El pipeline solo transforma las features. Si se desea inferencia sin supervivencia (solo risk scores), esas columnas pueden omitirse en la entrada.

---

### Etapa 2 — StandardScaler por modelo

Cada modelo de deep learning tiene su propio `StandardScaler` (preprocessor) guardado durante el entrenamiento. El pipeline anterior ya produjo las 64 columnas, y cada modelo selecciona sus features y las escala.

```python
EXCLUDE_MODEL = {'CIF', 'Start', 'Stop', 'Event'}

def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_MODEL]

def scale_for_model(df_full, preprocessor):
    feature_cols = get_feature_cols(df_full)
    return preprocessor.transform(df_full[feature_cols]).astype('float32')
```

---

## Estructura de Modelos Entrenados

```
notebooks/
├── cox_time/
│   ├── cox_time_preprocessor.joblib   → ColumnTransformer (StandardScaler)
│   ├── cox_time_model_weights.pt      → state_dict de la red
│   ├── cox_time_net.pt                → red completa serializada
│   ├── cox_time_labtrans.joblib       → label_transform de CoxTime
│   └── cox_time_baseline_hazards.joblib → hazard base acumulado
│
├── deepsurv_models/
│   ├── deepsurv_preprocessor.pkl
│   ├── deepsurv_model_weights.pt
│   └── deepsurv_baseline_hazards.joblib
│
├── cox_cc_models/
│   ├── cox_cc_preprocessor.pkl
│   ├── cox_cc_modelo_weights.pt
│   └── cox_cc_baseline_hazards.joblib
│
├── deephit_models/
│   ├── deephit_preprocessor.pkl
│   ├── deephit_model_weights.pt
│   ├── deephit_net.pt
│   └── deephit_labtrans.pkl
│
├── coxph_models/
│   ├── preprocessor.joblib
│   └── coxph_model.joblib
│
└── rsf_models/
     ├── rsf_preprocessor.pkl
     └── rsf_model.joblib
```

> **Nota:** Los artefactos están en `notebooks/<modelo>_models/`, no en `models/`. La carpeta `models/` contiene solo los artefactos del pipeline de preprocesamiento general.

---

## Función unificada de inferencia

Ejemplo completo que recibe datos crudos, aplica el pipeline general y luego el modelo específico:

```python
import joblib
import pandas as pd
import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxTime, CoxPH, CoxCC, DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime

def predict_survival(model_name, df_raw):
    """
    model_name: 'cox_time' | 'deepsurv' | 'cox_cc' | 'deephit' | 'coxph' | 'rsf'

    df_raw: DataFrame con columnas tipo raw_data.csv (N*, F*, categóricas)
    Retorna: (risk_scores, survival_curves)
    """
    # ── Etapa 1: Pipeline general ──
    df_processed = preprocess_raw_data(df_raw)

    # ── Etapa 2: Modelo específico ──
    if model_name == 'cox_time':
        return _predict_cox_time(df_processed)
    elif model_name == 'deepsurv':
        return _predict_deepsurv(df_processed)
    elif model_name == 'cox_cc':
        return _predict_cox_cc(df_processed)
    elif model_name == 'deephit':
        return _predict_deephit(df_processed)
    elif model_name == 'coxph':
        return _predict_coxph(df_processed)
    elif model_name == 'rsf':
        return _predict_rsf(df_processed)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
```

A continuación, el detalle de cada modelo.

---

## Datos de Entrada

Cada fila representa una empresa/entidad con:

| Columnas | Descripción |
|----------|-------------|
| `N1`..`N43` | Variables numéricas (razones financieras, indicadores) |
| `F16`..`F43` | Variables financieras adicionales |
| `Stop` | Tiempo observado (meses hasta evento o censura) |
| `Event` | Indicador (1 = evento ocurrió, 0 = censurado) |
| `CIF` | Identificador de la empresa (se ignora) |
| `Start` | Tiempo de inicio (se ignora) |

**Para inferencia**, solo necesitas las columnas de features (`N*`, `F*`). Las columnas `Stop`, `Event`, `CIF`, `Start` son opcionales — si están presentes se ignoran automáticamente.

---

## 1. Cox-Time (pycox)

```python
import joblib
import torch
import torchtuples as tt
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime

BASE = "notebooks/cox_time_models/"

preprocessor = joblib.load(f"{BASE}cox_time_preprocessor.joblib")
labtrans     = joblib.load(f"{BASE}cox_time_labtrans.joblib")
baseline_haz = joblib.load(f"{BASE}cox_time_baseline_hazards.joblib")

df_new = pd.read_csv("nuevos_datos.csv")
df_processed = preprocess_raw_data(df_new)
X = scale_for_model(df_processed, preprocessor)
in_features = X.shape[1]

net = MLPVanillaCoxTime(in_features, [128, 64, 32], batch_norm=True, dropout=0.2)
net.load_state_dict(torch.load(f"{BASE}cox_time_model_weights.pt"))
net.eval()

model = CoxTime(net, tt.optim.AdamWR, labtrans=labtrans)
model.baseline_hazards_ = baseline_haz
model.compute_baseline_hazards()

# Risk scores (mayor = peor)
risk = model.predict(X)

# Curvas de supervivencia: DataFrame (tiempo × muestra)
surv_df = model.predict_surv_df(X)
```

**Outputs:**
- `risk`: array `(n,)` — log-hazard parcial. Mayor valor = mayor riesgo.
- `surv_df`: DataFrame con shape `(n_times, n)`. Índice = tiempos de entrenamiento. Columnas = muestras. Valores = probabilidad de sobrevivir más allá de t.

---

## 2. DeepSurv (pycox)

```python
import joblib
import torch
import torchtuples as tt
from pycox.models import CoxPH

BASE = "notebooks/deepsurv_models/"

preprocessor = joblib.load(f"{BASE}deepsurv_preprocessor.pkl")
baseline_haz = joblib.load(f"{BASE}deepsurv_baseline_hazards.joblib")

df_new = pd.read_csv("nuevos_datos.csv")
df_processed = preprocess_raw_data(df_new)
X = scale_for_model(df_processed, preprocessor)
in_features = X.shape[1]

net = tt.practical.MLPVanilla(in_features, [128, 64, 32], 1,
                              batch_norm=True, dropout=0.2, output_bias=False)
net.load_state_dict(torch.load(f"{BASE}deepsurv_model_weights.pt"))
net.eval()

model = CoxPH(net, tt.optim.AdamWR)
model.baseline_hazards_ = baseline_haz

risk = model.predict(X)
surv_df = model.predict_surv_df(X)
```

**Outputs:** Misma interpretación que Cox-Time.

---

## 3. Cox-CC (pycox)

```python
import joblib
import torch
import torchtuples as tt
from pycox.models import CoxCC

BASE = "notebooks/cox_cc_models/"

preprocessor = joblib.load(f"{BASE}cox_cc_preprocessor.pkl")
baseline_haz = joblib.load(f"{BASE}cox_cc_baseline_hazards.joblib")

df_new = pd.read_csv("nuevos_datos.csv")
df_processed = preprocess_raw_data(df_new)
X = scale_for_model(df_processed, preprocessor)
in_features = X.shape[1]

net = tt.practical.MLPVanilla(in_features, [128, 64, 32], 1,
                              batch_norm=True, dropout=0.2, output_bias=False)
net.load_state_dict(torch.load(f"{BASE}cox_cc_modelo_weights.pt"))
net.eval()

model = CoxCC(net, tt.optim.AdamWR)
model.baseline_hazards_ = baseline_haz

risk = model.predict(X)
surv_df = model.predict_surv_df(X)
```

---

## 4. DeepHit (pycox)

```python
import joblib
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle

BASE = "notebooks/deephit_models/"

preprocessor = joblib.load(f"{BASE}deephit_preprocessor.pkl")
labtrans     = joblib.load(f"{BASE}deephit_labtrans.pkl")

df_new = pd.read_csv("nuevos_datos.csv")
df_processed = preprocess_raw_data(df_new)
X = scale_for_model(df_processed, preprocessor)
in_features = X.shape[1]

net = tt.practical.MLPVanilla(in_features, [128, 64, 32],
                              labtrans.out_features, batch_norm=True, dropout=0.2)
net.load_state_dict(torch.load(f"{BASE}deephit_model_weights.pt"))
net.eval()

model = DeepHitSingle(net, tt.optim.AdamWR, alpha=0.2, sigma=0.1,
                      duration_index=labtrans.cuts)

# Curvas suavizadas con interpolación lineal
surv_df = model.interpolate(10).predict_surv_df(X)

# Sin interpolación (escalones discretos):
# surv_df = model.predict_surv_df(X)
```

**Nota:** DeepHit no tiene `predict()` porque es un modelo de riesgos competitivos discretos. La curva de supervivencia es la salida principal.

---

## 5. CoxPH (lifelines)

```python
import joblib
import pandas as pd

BASE = "notebooks/coxph_models/"

preprocessor = joblib.load(f"{BASE}preprocessor.joblib")
cph          = joblib.load(f"{BASE}coxph_model.joblib")

df_new = pd.read_csv("nuevos_datos.csv")
df_processed = preprocess_raw_data(df_new)
X = scale_for_model(df_processed, preprocessor)
cols_after = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X, columns=cols_after)

# Partial hazard (mayor = peor)
risk = cph.predict_partial_hazard(X_df)

# Curvas de supervivencia: DataFrame (tiempo × muestra)
surv_df = cph.predict_survival_function(X_df)

# Linear predictor
lp = cph.predict_log_partial_hazard(X_df)

# Tiempo mediano predicho
median_time = cph.predict_percentile(X_df, p=0.5)
```

**Outputs:**
- `risk`: Series `(n,)` — hazard ratio exp(β·x). Mayor = mayor riesgo instantáneo.
- `surv_df`: DataFrame `(n_times, n)` — curvas de supervivencia.
- `lp`: Series `(n,)` — predictor lineal β·x.
- `median_time`: Series `(n,)` — tiempo estimado al 50% de supervivencia.

---

## 6. RSF — Random Survival Forest (scikit-survival)

```python
import joblib
import numpy as np

BASE = "notebooks/rsf_models/"

preprocessor = joblib.load(f"{BASE}rsf_preprocessor.pkl")
rsf          = joblib.load(f"{BASE}rsf_model.joblib")

df_new = pd.read_csv("nuevos_datos.csv")
df_processed = preprocess_raw_data(df_new)
X = scale_for_model(df_processed, preprocessor)

# Risk scores (mayor = peor)
risk = rsf.predict(X)

# Curvas de supervivencia: array (muestras × tiempos)
surv_array = rsf.predict_survival_function(X, return_array=True)

# Tiempos correspondientes a las columnas de surv_array
times = rsf.unique_times_

# Hazard acumulado
cum_hazard = rsf.predict_cumulative_hazard_function(X, return_array=True)
```

**Outputs:**
- `risk`: array `(n,)` — expected mortality. Mayor = mayor riesgo.
- `surv_array`: array `(n, n_times)` — probabilidad de supervivencia en `unique_times_`.
- `times`: array `(n_times,)` — grilla temporal del bosque.

---

## Resumen comparativo

| Modelo | Librería | Risk score | Curva de supervivencia | Requiere baseline_hazards | Requiere labtrans |
|--------|----------|------------|----------------------|--------------------------|-------------------|
| Cox-Time | pycox | `model.predict(X)` | `model.predict_surv_df(X)` | Sí | Sí |
| DeepSurv | pycox | `model.predict(X)` | `model.predict_surv_df(X)` | Sí | No |
| Cox-CC | pycox | `model.predict(X)` | `model.predict_surv_df(X)` | Sí | No |
| DeepHit | pycox | N/A | `model.interpolate(10).predict_surv_df(X)` | No | Sí |
| CoxPH | lifelines | `cph.predict_partial_hazard(X_df)` | `cph.predict_survival_function(X_df)` | No | No |
| RSF | scikit-survival | `rsf.predict(X)` | `rsf.predict_survival_function(X, return_array=True)` | No | No |

**Formato de salida de curvas de supervivencia:**

- **pycox** → `pd.DataFrame`. Filas = tiempos, columnas = muestras.
- **CoxPH** → `pd.DataFrame`. Filas = tiempos, columnas = muestras.
- **RSF** → `np.ndarray` de shape `(n_muestras, n_tiempos)`.

Para convertir RSF a formato homogéneo:

```python
surv_df = pd.DataFrame(surv_array.T, index=rsf.unique_times_)
```

---

## Artefactos del pipeline de preprocesamiento

Los artefactos en `models/preprocessing/` fueron generados durante el entrenamiento y **no deben re-entrenarse**:

| Artifact | Objeto |
|----------|--------|
| `mice_imputer.joblib` | `IterativeImputer(BayesianRidge(), max_iter=3)` |
| `one_hot_encoder.joblib` | `OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')` |
| `yeo_johnson_transformer.joblib` | `PowerTransformer(method='yeo-johnson', standardize=True)` |
| `ohe_columns.joblib` | Lista de 27 nombres de columnas OHE |
| `numerical_columns.joblib` | Lista de 33 columnas numéricas normalizadas |
| `mice_imputation_columns.joblib` | Misma lista que `numerical_columns` |
| `expected_columns.joblib` | Lista ordenada de las 64 columnas finales |
| `pipeline_metadata.json` | Parámetros de configuración del pipeline |
