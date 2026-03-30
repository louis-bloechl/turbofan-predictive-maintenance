import xgboost as xgb
import mlflow.xgboost
import optuna
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error

from src.data.loader import load_cmapss_data
from src.data.split import split_data
from src.features.temporal import build_features

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw"

def run_optimization():
    print("Starting Optuna hyperparameter search")

    df_raw = load_cmapss_data(str(DATA_DIR / "train_FD001.txt"))
    df_features = build_features(df_raw)
    train_df, test_df = split_data(df_features)

    drop_cols = ['engine_id', 'time_cycle', 'rul']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['rul'].clip(upper=125)

    X_val = test_df.drop(columns=drop_cols)
    y_val = test_df['rul'].clip(upper=125)

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5)
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, predictions))

    study = optuna.create_study(direction="minimize", study_name="XGBoost_Optimization")
    study.optimize(objective, n_trials=25)

    print(f"Best RMSE: {study.best_value:.2f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # Merge best params with the static requirements
    final_params = {**study.best_params, "objective": "reg:squarederror", "tree_method": "hist", "random_state": 42}
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_train, y_train)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Turbofan_Engine_RUL")

    with mlflow.start_run(run_name="Best_Optuna_Model") as run:
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)
        
        # Save model file to the artifact store
        mlflow.xgboost.log_model(final_model, "turbofan_xgboost_v1")
        
        print(f"Best model is: {run.info.run_id}")

if __name__ == "__main__":
    run_optimization()