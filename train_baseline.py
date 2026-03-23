import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

BRAAK_TO_GROUP = {'Braak II': 0, 'Braak III': 0, 'Braak IV': 1, 'Braak V': 2, 'Braak VI': 2}

def clean_data(df):
    df = df[df['Braak'] != 'Braak 0']
    
    y = df['Braak']
    df = df.drop(['Braak', 'Overall CAA Score', 'Last CASI Score'], axis=1)
    df['aad_grouped'] = pd.cut(
        df['Age at Death'],
        bins=[0, 79, 89, np.inf],
        labels=['70-79', '80-89', '90+'],
        right=True
    )
    
    df = pd.get_dummies(
        df,
        columns=[
            'Sex',
            'CERAD score',
            'aad_grouped'
        ],
        dtype=int
    )
    df = pd.get_dummies(
        df,
        columns=[
            'Atherosclerosis',
        ],
        dummy_na=True,
        dtype=int
    )
    df = df.rename({'Atherosclerosis_nan': 'Atherosclerosis_None'}, axis=1)
    df = df.drop('Age at Death', axis=1)
    return df, y


if __name__ == '__main__':
    
    df = pd.read_csv('trimmed.csv', index_col=0)
    x, y = clean_data(df)
    
    with open('folds.json', 'r') as f:
        splits = json.load(f)
    
    for split in splits:
        log_reg = LogisticRegression(random_state=2026, max_iter=450)
        log_reg.fit(x.loc[split['train']], y.loc[split['train']])   
        print(log_reg.score(x.loc[split['test']], y.loc[split['test']]))
        
    