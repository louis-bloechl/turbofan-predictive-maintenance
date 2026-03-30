import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split_data(df: pd.DataFrame, test_size = 0.2, random_state = 42):
    """
    Split dataset into train/test sets with engine-level isolation to
    prevent temporal data leakage
    """
    # GroupShuffleSplit makes sure that if engine is in test set, none of its history
    # is seen during training
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['engine_id']))

    # Discard shuffled indices and 0-index the rows
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    return train_df, test_df