"""
Script CLI para inferencia unificada con modelos de supervivencia.

Preprocesa datos crudos usando el pipeline de models/preprocessing/
y ejecuta predicción con el modelo seleccionado.

Uso:
    python scripts/run_inference.py --input ruta/datos.csv --model cox_time --output ./resultados
    python scripts/run_inference.py --input ruta/datos.csv --model coxph --output ./resultados --no-preprocess
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchtuples as tt

try:
    from preprocessing_pipeline import preprocess_raw_data, FEATURE_EXCLUDE
except ImportError:
    from scripts.preprocessing_pipeline import preprocess_raw_data, FEATURE_EXCLUDE

# ── Rutas base de los modelos ──────────────────────────────────────────
MODEL_DIRS = {
    "cox_time": Path("notebooks/cox_time_models/"),
    "deepsurv": Path("notebooks/deepsurv_models/"),
    "cox_cc": Path("notebooks/cox_cc_models/"),
    "deephit": Path("notebooks/deephit_models/"),
    "coxph": Path("notebooks/coxph_models/"),
    "rsf": Path("notebooks/rsf_models/"),
}

# ── Arquitecturas de red (deben coincidir con el entrenamiento) ────────
try:
    from pycox.models import CoxPH, CoxCC, CoxTime, DeepHitSingle
    from pycox.models.cox_time import MLPVanillaCoxTime

    HAS_PYCOX = True
except ImportError:
    HAS_PYCOX = False

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

try:
    from sksurv.ensemble import RandomSurvivalForest
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


def _scale_for_model(df, preprocessor):
    """Aplica el StandardScaler específico del modelo al DataFrame preprocesado."""
    feature_cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    model_cols = [c for c in feature_cols if c in preprocessor.feature_names_in_]
    if len(model_cols) != len(preprocessor.feature_names_in_):
        expected = set(preprocessor.feature_names_in_)
        got = set(feature_cols)
        missing_expected = expected - got
        msg = (
            f"El preprocesador del modelo espera {len(expected)} columnas, "
            f"pero el DataFrame preprocesado tiene {len(got)} columnas de features. "
            f"Faltan: {sorted(missing_expected)[:5]}..."
        )
        raise ValueError(msg)
    return preprocessor.transform(df[feature_cols]).astype("float32")


# ── Predictores específicos ────────────────────────────────────────────

def predict_cox_time(df, base_dir):
    preprocessor = joblib.load(base_dir / "cox_time_preprocessor.joblib")
    labtrans = joblib.load(base_dir / "cox_time_labtrans.joblib")
    baseline_haz = joblib.load(base_dir / "cox_time_baseline_hazards.joblib")

    X = _scale_for_model(df, preprocessor)
    in_features = X.shape[1]

    net = MLPVanillaCoxTime(in_features, [128, 64, 32], batch_norm=True, dropout=0.2)
    net.load_state_dict(torch.load(base_dir / "cox_time_model_weights.pt", weights_only=True))
    net.eval()

    model = CoxTime(net, tt.optim.AdamWR, labtrans=labtrans)
    model.baseline_hazards_ = baseline_haz
    model.compute_baseline_hazards()

    risk = model.predict(X)
    surv_df = model.predict_surv_df(X)
    return risk, surv_df


def predict_deepsurv(df, base_dir):
    preprocessor = joblib.load(base_dir / "deppsurv_preprocessor.pkl")
    baseline_haz = joblib.load(base_dir / "deppsurv_baseline_hazards.joblib")

    X = _scale_for_model(df, preprocessor)
    in_features = X.shape[1]

    net = tt.practical.MLPVanilla(in_features, [128, 64, 32], 1,
                                  batch_norm=True, dropout=0.2, output_bias=False)
    net.load_state_dict(torch.load(base_dir / "deppsurv_modelo_weights.pt", weights_only=True))
    net.eval()

    model = CoxPH(net, tt.optim.AdamWR)
    model.baseline_hazards_ = baseline_haz

    risk = model.predict(X)
    surv_df = model.predict_surv_df(X)
    return risk, surv_df


def predict_cox_cc(df, base_dir):
    preprocessor = joblib.load(base_dir / "cox_cc_preprocessor.pkl")
    baseline_haz = joblib.load(base_dir / "cox_cc_baseline_hazards.joblib")

    X = _scale_for_model(df, preprocessor)
    in_features = X.shape[1]

    net = tt.practical.MLPVanilla(in_features, [128, 64, 32], 1,
                                  batch_norm=True, dropout=0.2, output_bias=False)
    net.load_state_dict(torch.load(base_dir / "cox_cc_modelo_weights.pt", weights_only=True))
    net.eval()

    model = CoxCC(net, tt.optim.AdamWR)
    model.baseline_hazards_ = baseline_haz

    risk = model.predict(X)
    surv_df = model.predict_surv_df(X)
    return risk, surv_df


def predict_deephit(df, base_dir):
    preprocessor = joblib.load(base_dir / "deephit_preprocessor.pkl")
    labtrans = joblib.load(base_dir / "deephit_labtrans.pkl")

    X = _scale_for_model(df, preprocessor)
    in_features = X.shape[1]

    net = tt.practical.MLPVanilla(in_features, [128, 64, 32],
                                  labtrans.out_features, batch_norm=True, dropout=0.2)
    net.load_state_dict(torch.load(base_dir / "deephit_model_weights.pt", weights_only=True))
    net.eval()

    model = DeepHitSingle(net, tt.optim.AdamWR, alpha=0.2, sigma=0.1,
                          duration_index=labtrans.cuts)
    surv_df = model.interpolate(10).predict_surv_df(X)
    return None, surv_df


def predict_coxph(df, base_dir):
    preprocessor = joblib.load(base_dir / "preprocessor.joblib")
    cph = joblib.load(base_dir / "coxph_model.joblib")

    X = _scale_for_model(df, preprocessor)
    cols_after = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X, columns=cols_after)

    risk = cph.predict_partial_hazard(X_df)
    surv_df = cph.predict_survival_function(X_df)
    lp = cph.predict_log_partial_hazard(X_df)
    median_time = cph.predict_percentile(X_df, p=0.5)
    return risk, surv_df


def predict_rsf(df, base_dir):
    preprocessor = joblib.load(base_dir / "rsf_preprocessor.pkl")
    rsf = joblib.load(base_dir / "rsf_model.joblib")

    X = _scale_for_model(df, preprocessor)

    risk = rsf.predict(X)
    surv_array = rsf.predict_survival_function(X, return_array=True)
    surv_df = pd.DataFrame(surv_array.T, index=rsf.unique_times_)
    return risk, surv_df


# ── Registro de modelos ────────────────────────────────────────────────

PREDICTORS = {
    "cox_time": predict_cox_time,
    "deepsurv": predict_deepsurv,
    "cox_cc": predict_cox_cc,
    "deephit": predict_deephit,
    "coxph": predict_coxph,
    "rsf": predict_rsf,
}

MODEL_REQUIREMENTS = {
    "cox_time": ["pycox"],
    "deepsurv": ["pycox"],
    "cox_cc": ["pycox"],
    "deephit": ["pycox"],
    "coxph": ["lifelines"],
    "rsf": ["sksurv"],
}


def check_dependencies(model_name):
    deps = MODEL_REQUIREMENTS.get(model_name, [])
    for dep in deps:
        if dep == "pycox" and not HAS_PYCOX:
            raise ImportError("pycox no está instalado. Ejecuta: pip install pycox")
        if dep == "lifelines" and not HAS_LIFELINES:
            raise ImportError("lifelines no está instalado. Ejecuta: pip install lifelines")
        if dep == "sksurv" and not HAS_SKSURV:
            raise ImportError("scikit-survival no está instalado. Ejecuta: pip install scikit-survival")


def check_model_artifacts(model_name):
    base_dir = MODEL_DIRS[model_name]
    if not base_dir.exists():
        return False, f"El directorio del modelo no existe: {base_dir}"
    predictor = PREDICTORS[model_name]
    # Verificar que los archivos existan inspeccionando el código de la función
    return True, None


# ── Función principal ──────────────────────────────────────────────────

def run_inference(input_path, model_name, output_dir, skip_preprocess=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    print(f"[1/4] Cargando datos: {input_path}")
    df_input = pd.read_csv(input_path)
    print(f"       Filas: {df_input.shape[0]}, Columnas: {df_input.shape[1]}")

    # 2. Preprocesar
    if skip_preprocess:
        print(f"[2/4] Usando datos sin preprocesamiento del pipeline general")
        df_processed = df_input
    else:
        print(f"[2/4] Ejecutando pipeline de preprocesamiento (MICE + OHE + Yeo-Johnson)")
        df_processed = preprocess_raw_data(df_input)
        print(f"       Features generadas: {df_processed.shape[1]}")

    # 3. Verificar modelo
    print(f"[3/4] Cargando modelo: {model_name}")
    check_dependencies(model_name)

    base_dir = MODEL_DIRS[model_name]
    if not base_dir.exists():
        print(f"  ⚠  El directorio del modelo no existe: {base_dir}")
        print(f"     Debes entrenar el modelo '{model_name}' primero.")
        print(f"     Ejecuta: python scripts/train_{model_name}.py")
        sys.exit(1)

    predictor = PREDICTORS[model_name]

    # 4. Predecir
    print(f"[4/4] Ejecutando predicción...")
    try:
        result = predictor(df_processed, base_dir)
    except ValueError as e:
        print(f"  ✗ Error de compatibilidad: {e}")
        print(f"\nSugerencia: El modelo '{model_name}' fue entrenado con un formato")
        print(f"de datos diferente. Es posible que necesites reentrenarlo con el")
        print(f"nuevo pipeline. Ejecuta: python scripts/train_{model_name}.py\n")
        sys.exit(1)

    risk, surv_df = result if len(result) == 2 else (result[0], result[1])

    # 5. Guardar resultados
    risk_out = output_dir / "risk_scores.csv"
    surv_out = output_dir / "survival_curves.csv"

    pd.Series(risk, name="risk_score").to_csv(risk_out, index=False)
    surv_df.to_csv(surv_out)
    print(f"\n✅ Resultados guardados en: {output_dir}/")
    print(f"   - risk_scores.csv        ({risk_out.stat().st_size / 1024:.1f} KB)")
    print(f"   - survival_curves.csv    ({surv_out.stat().st_size / 1024:.1f} KB)")
    print(f"\nRisk scores (primeras 5): {np.round(risk[:5], 4).tolist()}")
    print(f"Survival curves: {surv_df.shape[0]} tiempos × {surv_df.shape[1]} muestras")


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia unificada con modelos de supervivencia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python scripts/run_inference.py --input datos.csv --model cox_time\n"
            "  python scripts/run_inference.py --input datos.csv --model coxph -o ./resultados\n"
            "\nModelos disponibles: cox_time, deepsurv, cox_cc, deephit, coxph, rsf\n"
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="Ruta al CSV con datos crudos")
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(PREDICTORS.keys()),
        help="Modelo a usar para la predicción",
    )
    parser.add_argument(
        "--output", "-o",
        default="./resultados",
        help="Directorio donde guardar los resultados (default: ./resultados)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Omitir el pipeline de preprocesamiento (esperar datos ya preprocesados)",
    )
    args = parser.parse_args()
    run_inference(args.input, args.model, args.output, args.no_preprocess)


if __name__ == "__main__":
    main()
