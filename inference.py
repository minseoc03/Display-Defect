import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def inference(X, y, test, params, seeds = [1, 2, 3, 4, 5]):
    test['target'] = 0

    for seed in seeds:
        xgb = XGBClassifier(
            **params,
            scale_pos_weight=8.5,
            seed=seed
        )
        xgb.fit(X, y)
        test['target'] += xgb.predict_proba(test)[:, 1]

    test['target'] = test['target'] / len(seeds)
    test['target'] = test['target'] >= 0.5
    test['target'] = test['target'].replace(True, 'AbNormal')
    test['target'] = test['target'].replace(False, 'Normal')
    
    return test