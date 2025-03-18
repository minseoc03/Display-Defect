import numpy as np
import pandas as pd
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
from xgboost import XGBClassifier

def optimize(X, y, seed):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = seed)

    sampler = TPESampler(seed =seed)

    def objective(trial):

        cbrm_param = {
            'n_estimators':trial.suggest_int('n_estimators', 10, 1000),
            'learning_rate' : trial.suggest_uniform('learning_rate',0.01, 1),
            'lambda': trial.suggest_uniform('lambda',1e-5,100),
            'alpha': trial.suggest_uniform('alpha',1e-5,100),
            'subsample': trial.suggest_uniform('subsample',0,1),
            'max_depth': trial.suggest_int('max_depth',1, 15),
            'min_child_weight': trial.suggest_int('min_child_weight',1,30),
            'gamma': trial.suggest_loguniform("gamma", 0.1, 1.0),
            'colsample_bytree':trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'colsample_bylevel':trial.suggest_float('colsample_bylevel', 0.4, 1.0),
            'colsample_bynode':trial.suggest_float('colsample_bynode', 0.4, 1.0),
            'scale_pos_weight': 8.5
        }

        model_xgb = XGBClassifier(**cbrm_param, early_stopping_rounds=25)
        model_xgb = model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                            verbose=0)

        f1 = f1_score(y_val, model_xgb.predict(X_val))
        return f1

    optuna_xgb = optuna.create_study(direction='maximize', sampler=sampler)
    optuna_xgb.optimize(objective, n_trials = 1000)

    xgb_trial = optuna_xgb.best_trial
    xgb_trial_params = xgb_trial.params
    print('Best Trial: score {},\nparams {}'.format(xgb_trial.value, xgb_trial_params))

    return xgb_trial, xgb_trial_params