# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from typing import Tuple


import numpy as np
import pandas as pd
import warnings


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


from imblearn.over_sampling import SMOTE

import joblib


# %%
df = pd.read_csv(
    'https://raw.githubusercontent.com/pedro-otl/tc_4/refs/heads/main/Obesity_1.csv', sep=',')

# %%
df

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.columns

# %%
df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Height': 'height', 'Weight': 'weight', 'FAVC': 'favc', 'FCVC': 'fcvc', 'NCP': 'ncp',
                        'CAEC': 'caec', 'SMOKE': 'smoke', 'CH2O': 'ch2o', 'SCC': 'scc', 'FAF': 'faf', 'TUE': 'tue', 'CALC': 'calc', 'MTRANS': 'mtrans', 'Obesity': 'obesity'})

# %%
df["fcvc"] = df["fcvc"].round().astype(int)
df["ncp"] = df["ncp"].round().astype(int)
df["ch2o"] = df["ch2o"].round().astype(int)
df["faf"] = df["faf"].round().astype(int)
df["tue"] = df["tue"].round().astype(int)

# %%
df_treino, df_teste = train_test_split(df, test_size=0.2, random_state=42)

# %%
df_treino.shape

# %%
df_treino_copy = df_treino.copy()
df_teste_copy = df_teste.copy()

# %%
df_treino.to_csv('train.csv', index=False)
df_teste.to_csv('test.csv', index=False)

# %%


class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=['height', 'weight', 'age', 'fcvc', 'ncp', 'ch2o', 'faf', 'tue']):
        self.min_max_scaler = min_max_scaler

    def fit(self, df):
        return self

    def transform(self, df):
        min_max_enc = MinMaxScaler()
        df[self.min_max_scaler] = min_max_enc.fit_transform(
            df[self.min_max_scaler])
        return df

# %%


class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['mtrans']):
        self.OneHotEncoding = OneHotEncoding

    def fit(self, df):
        return self

    def transform(self, df):
        if (set(self.OneHotEncoding).issubset(df.columns)):
            def one_hot_enc(df, OneHotEncoding):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[OneHotEncoding])
                feature_names = one_hot_enc.get_feature_names_out(
                    OneHotEncoding)
                df = pd.DataFrame(one_hot_enc.transform(df[self.OneHotEncoding]).toarray(),
                                  columns=feature_names, index=df.index)

                return df

            def concat_with_rest(df, one_hot_enc_df, OneHotEncoding):
                outras_features = [
                    feature for feature in df.columns if feature not in OneHotEncoding]
                df_concat = pd.concat(
                    [one_hot_enc_df, df[outras_features]], axis=1)
                return df_concat

            df_OneHotEncoding = one_hot_enc(df, self.OneHotEncoding)

            df_full = concat_with_rest(
                df, df_OneHotEncoding, self.OneHotEncoding)
            return df_full
        else:
            print("Uma ou mais features não estão no data frame")
            return df

# %%


class ObesityOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ["caec", "calc"]
        self.categories = [
            ["no", "Sometimes", "Frequently", "Always"],
            ["no", "Sometimes", "Frequently", "Always"]
        ]
        self.encoder = OrdinalEncoder(categories=self.categories)

    def fit(self, df, y=None):
        self.encoder.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.copy()
        df[self.columns] = self.encoder.transform(df[self.columns])
        return df

# %%


class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or [
            "gender", "family_history", "favc", "smoke", "scc"
        ]
        self.label_encoders = {}

    def fit(self, df, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le
        return self

    def transform(self, df):
        df = df.copy()
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        return df

# %%


def pipeline(df):
    pipeline = Pipeline([
        ('OneHotEncoder', OneHotEncodingNames()),
        ('OrdinalEncoder', ObesityOrdinalEncoder()),
        ('BinaryEncoder', BinaryLabelEncoder()),
        ('MinMaxScaler', MinMax())])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline


# %%
treino = pipeline(df_treino)

# %%
treino.columns

# %%
treino.head()

# %%
X_treino, y_treino = treino.loc[:,
                                treino.columns != 'obesity'], treino['obesity']

# %%
teste = pipeline(df_teste)

# %%
X_teste, y_teste = teste.loc[:, teste.columns != 'obesity'], teste['obesity']

# %%
param_grid_rf = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_depth': [2, 3, 4, 5, 6, 7], 'min_samples_split': [10, 20, 0.05, 0.10],
                 'min_samples_leaf': [5, 10, 20, 0.01, 0.02], 'max_features': ['sqrt', 'log2', None], 'random_state': [7],
                 'class_weight': [None, 'balanced']}
gs_metric_rf = make_scorer(accuracy_score, greater_is_better=True)

grid_rf = GridSearchCV(RandomForestClassifier(
), param_grid=param_grid_rf, scoring=gs_metric_rf, cv=5, n_jobs=-1, verbose=3)

grid_rf.fit(X_treino, y_treino)

rf_params = grid_rf.best_params_

print('RF: ', rf_params)

# %%
rf = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                            max_depth=7, max_features=None, min_samples_leaf=5, min_samples_split=10, random_state=7)

rf.fit(X_treino, y_treino)

rf_pred = rf.predict(X_teste)

accuracy_rf = rf.score(X_teste, y_teste)
print('RF: ', accuracy_rf)


# %%
joblib.dump(rf, 'rf.joblib')
