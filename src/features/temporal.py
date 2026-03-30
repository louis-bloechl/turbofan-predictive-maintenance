import pandas as pd

# Settings and sensors with no change in the FD001 engine dataset
constant_sensors = [
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_5', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'
]

def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sensor columns that stay constant to reduce noise-to-signal ratio
    and improve model focus on useful signals
    """
    cols_to_drop = [col for col in constant_sensors if col in df.columns]
    return df.drop(columns=cols_to_drop)

def add_rolling_features(df: pd.DataFrame, window_sizes: list[int] = [10, 20]) -> pd.DataFrame:
        """
        Calculate rolling mean and std dev for sensor readings to
        capture temporal degradation trends for each engine
        """
        df_out = df.copy()
        sensor_cols = [col for col in df_out.columns if col.startswith('sensor_')]

        # Group by engine to prevent temporal leakage between different engines
        grouped = df_out.groupby('engine_id')

        # Calculate mean and std dev based on 10 and 20 row/observation windows
        # to capture macro and micro trends
        for window in window_sizes:
            for col in sensor_cols:
                # Mean captures macro trends for each sensor
                # Ex. temperature rising over 10 cycles -> engine degrading
                df_out[f'{col}_roll_mean_{window}'] = grouped[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

                # Standard deviation captures micro trends for each sensor
                # Ex. sensor readings bouncing around unstably -> engine degrading
                df_out[f'{col}_roll_std_{window}'] = grouped[col].transform(
                    # fillna covers NaN with 0 in the case of one row
                    lambda x: x.rolling(window, min_periods=1).std().fillna(0)
                )
        
        return df_out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
     """
     Pipeline execution for feature engineering
     """
     df_cleaned = drop_constant_features(df)
     df_engineered = add_rolling_features(df_cleaned)
     return df_engineered.dropna().reset_index(drop=True)