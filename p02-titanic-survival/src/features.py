import pandas as pd


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    for df in [train_df, test_df]:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
             'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

        df['AgeBand']  = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                                 labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
        # Using hardcoded bins derived from original titanic qcut (0, 7.91, 14.454, 31.0, 512.329)
        df['FareBand'] = pd.cut(df['Fare'], bins=[-0.001, 7.91, 14.454, 31.0, 600.0],
                                  labels=['Low', 'Mid', 'High', 'VeryHigh'])
        df['HasCabin'] = df['Cabin'].notna().astype(int)

    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    test_df = test_df.drop(columns=drop_cols, errors='ignore')

    print(f"\nAfter feature engineering — Train: {train_df.shape}")
    return train_df, test_df
