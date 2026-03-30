import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

from src.data.loader import load_cmapss_data
from src.features.temporal import build_features

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "app" / "model_weights"

def s_score(y_true, y_pred):
    """
    NASA S-score evaluates how accurate an algorithm is at predicting RUL
    Penalizes late predictions (dangerous) more than early predictions (safe)
    """
    diff = y_pred - y_true
    s_score = np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    return s_score

def load_inference_model(model_path=None):
    """
    Load serialized XGboost model for inference
    """
    if model_path is None:
        model_path = MODEL_DIR / "model.ubj"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weight file not found at {model_path}")
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    return model

def get_engine_analytics(engine_data, model):
    """
    Decoupled logic to compute RUL prediction and XAI attributions
    """
    # Isolate inference features from metadata
    drop_cols = ['engine_id', 'time_cycle', 'rul', 'predicted_rul', 'status']
    X_predict = engine_data.drop(columns=drop_cols, errors='ignore')

    # Transform data to optimized format
    dmatrix = xgb.DMatrix(X_predict)
    predicted_rul = int(model.predict(X_predict)[0])
    # Compute marginal feature attributions (SHAP values)
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)[0]

    # Prioritize features by absolute magnitude
    df_contributions = pd.DataFrame({
        'feature': X_predict.columns.tolist(),
        'contribution': contributions[:-1]
    })
    df_contributions['abs_contribution'] = df_contributions['contribution'].abs()
    df_contributions = df_contributions.sort_values(by='abs_contribution', ascending=False)

    return predicted_rul, df_contributions, contributions

def run_evaluation():
    """
    CLI utility to evaluate model performance against ground truth labels
    """
    try:
        model = load_inference_model()
    except FileNotFoundError as e:
        print("Error: Please run 'python src/models/train.py' first to generate the model weights")
        return

    # Load blind test data and build temporal features
    test_file = DATA_DIR / "test_FD001.txt"
    label_file = DATA_DIR / "RUL_FD001.txt"
    df_test_raw = load_cmapss_data(str(test_file))
    df_test_features = build_features(df_test_raw)

    # Load ground truth labels
    try:
        y_true = pd.read_csv(label_file, sep=r'\s+', header=None, names=['actual_rul'])
        y_true['engine_id'] = y_true.index + 1
    except FileNotFoundError:
        print(f"Ground truth labels not found at {label_file}")
        return

    # Assess performance on final cycle of each engine
    last_cycle_df = df_test_features.groupby('engine_id').tail(1).copy()
    X_eval = last_cycle_df.drop(columns=['engine_id', 'time_cycle', 'rul'], errors='ignore')

    last_cycle_df['predicted_rul'] = model.predict(X_eval)
    results_df = pd.merge(last_cycle_df[['engine_id', 'predicted_rul']], y_true, on='engine_id')

    # Calculate fleet-wide metrics
    rmse = np.sqrt(mean_squared_error(results_df['actual_rul'], results_df['predicted_rul']))
    mae = mean_absolute_error(results_df['actual_rul'], results_df['predicted_rul'])
    score = s_score(results_df['actual_rul'], results_df['predicted_rul'])

    print(f"Blind Test RMSE: {rmse:.2f} cycles")
    print(f"Blind Test MAE: {mae:.2f} cycles")
    print(f"NASA S-Score: {score:.2f}\n")

if __name__ == "__main__":
    run_evaluation()