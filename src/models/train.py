import xgboost as xgb
import mlflow.xgboost
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from src.data.loader import load_cmapss_data
from src.data.split import split_data
from src.features.temporal import build_features

BASE_DIR = Path(__file__).resolve().parents[2]
DEPLOY_PATH = BASE_DIR / "app" / "model_weights"
DATA_DIR = BASE_DIR / "data" / "raw"

def train_model():
    # Initialize MLflow experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Turbofan_Engine_RUL")
    with mlflow.start_run(run_name="turbofan_xgboost_run_v1"):
        print("Starting training pipeline")

        # Use FD001 as training set
        df_raw = load_cmapss_data(str(DATA_DIR / "train_FD001.txt"))
        df_features = build_features(df_raw)
        train_df, test_df = split_data(df_features)

        # Drop metadata
        drop_cols = ['engine_id', 'time_cycle', 'rul']
        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df['rul'].clip(upper=125)
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df['rul'].clip(upper=125)

        # Set hyperparameters
        # Parameters below were chosen based off most successful Optuna runs
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist", # optimize for speed on tabular data
            "random_state": 42,
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.7, # row sampling
            "colsample_bytree": 0.7, # feature sampling
            "min_child_weight": 5 # noise filter
        }
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluation
        # RMSE -> on average, how many cycles off
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        print(f"Pipeline results: RMSE = {rmse:.2f} and MAE: {mae:.2f}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Feauture importance logging
        plt.figure(figsize=(10, 12))
        xgb.plot_importance(model, max_num_features=10, importance_type='weight')
        plt.title("Top 10 Predictive Features (Sensors/Windows)")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # Save MLflow flavor
        mlflow.xgboost.log_model(model, "turbofan_xgboost_v1")

        # Save model in location visible to dashboard
        DEPLOY_PATH.mkdir(parents=True, exist_ok=True)
        model.save_model(DEPLOY_PATH / "model.ubj")
        print(f"Model weights successfully saved at {DEPLOY_PATH}")


if __name__ == "__main__":
    train_model()