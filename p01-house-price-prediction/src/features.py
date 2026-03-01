import pandas as pd


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    for df in [train_df, test_df]:
        df['HouseAge']     = df['YrSold'] - df['YearBuilt']
        df['RemodAge']     = df['YrSold'] - df['YearRemodAdd']
        df['TotalSF']      = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalBaths']   = (df['FullBath'] + df['BsmtFullBath'] +
                              0.5 * df['HalfBath'] + 0.5 * df['BsmtHalfBath'])
        df['HasGarage']    = (df['GarageArea'] > 0).astype(int)
        df['HasPool']      = (df['PoolArea'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    print(f"\nAfter feature engineering — Train: {train_df.shape}")
    return train_df, test_df
