import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import xgboost as xgb
from xgboost import XGBClassifier

def train(X, y, params, seed):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state = seed)
    best_params = params
    f1_train = []
    f1_score_test = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = XGBClassifier(**best_params, scale_pos_weight = 8.5)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        f1_train.append(f1_score(y_train, y_train_pred))
        f1_score_test.append(f1_score(y_test, y_test_pred))
    f1_train = np.array(f1_train)
    f1_score_test = np.array(f1_score_test)
    print("mean f1 is: {}".format(f1_score_test.mean()))

    return model, f1_score_test.mean()