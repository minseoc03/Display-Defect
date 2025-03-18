import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import category_encoders as ce

def preprocess(PATH):
    """
    EDA / Pre-processing Part
    """
    #read data csv file
    train = pd.read_csv(f'{PATH}/train.csv')
    test = pd.read_csv(f'{PATH}/test.csv')

    #remove missing values
    train = train.dropna(how = 'all', axis = 1)

    #remove constant columns
    cols_to_drop = []

    for col in train.columns:
        if train[col].nunique() == 1:
            cols_to_drop.append(col)

    train = train.drop(cols_to_drop, axis = 1)

    #remove columns with missing values
    cols_with_null = []

    for col in train.columns:
        if train[col].isnull().sum() > 0:
            cols_with_null.append(col)

    train[cols_with_null] = train[cols_with_null].replace('OK', np.nan)
    test[cols_with_null] = test[cols_with_null].replace('OK', np.nan)

    train = train.drop(cols_with_null, axis = 1)
    test = test.drop(cols_with_null, axis = 1)

    #select categorical variables
    #conditions : 10 or less unique values OR object data types
    cat_col = []

    for col in train.columns:
        if train[col].nunique() <= 10 or train[col].dtypes == 'object':
            cat_col.append(col)

    train[cat_col] = train[cat_col].astype('str')
    test[cat_col] = test[cat_col].astype('str')

    #find highly skewed columns
    num_col = []
    for col in train.columns:
        if col not in cat_col:
            num_col.append(col)
    skewed_columns = train[num_col].apply(lambda x: x.skew()).sort_values(ascending=False)

    #calibrate skewness by using yeo-johnson PowerTransformer
    threshold = 0.75
    highly_skewed = skewed_columns[abs(skewed_columns) > threshold]
    train[highly_skewed] /= 100
    test[highly_skewed] /= 100

    pt = PowerTransformer(method = 'yeo-johnson',
                      standardize=False,
                      copy = False)

    pt.fit(train[highly_skewed.index])
    train[highly_skewed.index] = pt.transform(train[highly_skewed.index])
    test[highly_skewed.index] = pt.transform(test[highly_skewed.index])

    #numeric vairable scaling
    scaler = StandardScaler()
    scaler.fit(train[num_col])
    train[num_col] = scaler.transform(train[num_col])
    test[num_col] = scaler.transform(test[num_col])

    #encoding categorical variables
    low_cat_col = []
    high_cat_col = []

    for col in cat_col:
        if train[col].nunique() <= 5:
            low_cat_col.append(col)
        else:
            high_cat_col.append(col)
    
    #one-hot encoding variables with less than 5 unique values
    encoder = ce.OneHotEncoder(cols = low_cat_col)
    encoder.fit(train[low_cat_col])
    train_ohe = encoder.transform(train[low_cat_col])
    test_ohe = encoder.transform(test[low_cat_col])

    train = pd.concat([train, train_ohe], axis = 1)
    test = pd.concat([test, test_ohe], axis = 1)

    train = train.drop(low_cat_col, axis = 1)
    test = test.drop(low_cat_col, axis = 1)

    #WOE Encoder for variables with more than 5 unique values
    encoder = ce.WOEEncoder(cols = high_cat_col)
    encoder.fit(train[high_cat_col], train)
    train[high_cat_col] = encoder.transform(train[high_cat_col])
    test[high_cat_col] = encoder.transform(test[high_cat_col])

    X = train.drop('target', axis = 1)
    y = train['target']

    y = y.replace('Normal', 0)
    y = y.replace('AbNormal', 1)

    return X, y, test