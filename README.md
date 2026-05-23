# CDM Prototype — Predicción de Insolvencia en PYMEs Españolas

Prototipo de **Credit Default Model** para la predicción de insolvencia/quiebra en pequeñas y medianas empresas españolas mediante modelos de **supervivencia (survival analysis)**.

## Características

- **Pipeline de preprocesamiento**: imputación MICE, codificación one-hot y normalización Yeo-Johnson sobre datos financieros brutos.
- **6 modelos de supervivencia**: CoxPH (lifelines), DeepSurv, Cox-CC, Cox-Time, DeepHit (pycox/PyTorch) y Random Survival Forest (scikit-survival).
- **Interfaz interactiva** con Streamlit para cargar datos, ejecutar predicciones y visualizar curvas de supervivencia.
- **Estrategias de ensamble**: promedio de funciones de supervivencia, ponderado por rendimiento, basado en rangos, promedio bayesiano, stacking y consenso por votación.
- **CLI** para inferencia por lotes sin interfaz gráfica.

## Tecnologías

Python 3.10+ · Streamlit · PyTorch · pycox · lifelines · scikit-survival · scikit-learn · pandas · numpy · matplotlib · seaborn · joblib · scipy

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Interfaz gráfica (Streamlit)

```bash
streamlit run prototype_ui.py
```

Permite subir datos brutos (CSV/Excel), aplicar el pipeline, seleccionar modelos y ejecutar predicciones individuales o por ensamble.

### CLI — Inferencia por lotes

```bash
python scripts/run_inference.py --input datos.csv --model cox_time --output ./resultados
```

Modelos disponibles: `cox_time`, `deepsurv`, `cox_cc`, `deephit`, `coxph`, `rsf`.

### Entrenamiento

Cada modelo tiene su propio script en `scripts/`:

```bash
python scripts/train_coxph.py
python scripts/train_deepsurv.py
python scripts/train_cox_cc.py
python scripts/train_cox_time.py
python scripts/train_deephit.py
python scripts/train_rsf.py
```

## Estructura del proyecto

```
├── prototype_ui.py          # Aplicación Streamlit principal
├── settings.py              # Clases de modelos (DeepSurv, CoxPH, etc.)
├── assemblies.py            # Estrategias de ensamble
├── requirements.txt
│
├── scripts/
│   ├── preprocessing_pipeline.py   # Pipeline de preprocesado
│   ├── run_inference.py            # CLI de inferencia unificada
│   ├── utils.py                    # Utilidades compartidas
│   └── train_*.py                  # Scripts de entrenamiento (6 modelos)
│
├── datasets/                # Datos de entrenamiento, prueba y evaluación
├── models/preprocessing/    # Artefactos serializados del pipeline
├── notebooks/               # Modelos entrenados y cuadernos de exploración
└── docs/                    # Guías detalladas de inferencia y preprocesado
```

## Documentación

- `docs/inference_guide.md` — instrucciones detalladas de inferencia para cada modelo.
- `docs/data_preprocessing.md` — descripción completa del pipeline de preprocesado.
