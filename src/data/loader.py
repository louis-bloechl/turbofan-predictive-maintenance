import pandas as pd

index_columns = ['engine_id', 'time_cycle']
setting_columns = ['setting_1', 'setting_2', 'setting_3']
sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

all_columns = index_columns + setting_columns + sensor_columns

def load_cmapss_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=r'\s+', # treat spaces as boundaries between columns
        header=None,
        names=all_columns
    )

    # Find max cycle for each group
    max_cycles = df.groupby('engine_id')['time_cycle'].transform('max')
    # RUL = distance from current cycle to failure cycle (max)
    df['rul'] = max_cycles - df['time_cycle']

    return df