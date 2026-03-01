import streamlit as st
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

from sklearn.model_selection import GridSearchCV
from tc_4_cod_final import OneHotEncodingNames
from tc_4_cod_final import ObesityOrdinalEncoder
from tc_4_cod_final import BinaryLabelEncoder
from tc_4_cod_final import MinMax
from tc_4_cod_final import pipeline

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


from imblearn.over_sampling import SMOTE

import joblib
from joblib import load

st.write('# Estimador de Obesidade')

dados = pd.read_csv(
    'https://raw.githubusercontent.com/pedro-otl/tc_4/refs/heads/main/obesity_clean.csv', sep=',')


input_gender = str(st.selectbox('What is your gender?', ['Male', 'Female']))
input_age = int(st.number_input('What is your age?', 5, 130))
input_height = float(st.number_input(
    'What is your height? (in meters)', 0.3, 2.5))
input_weight = int(st.number_input('What is your weight? (in Kg)', 20, 550))
input_family_hist = str(st.selectbox(
    'Do any of your first-degree relatives (parents or siblings) have or have had overweight or obesity?', ['yes', 'no']))
input_favc = str(st.selectbox(
    'Do you frequently consume high-calorie foods (e.g., fast food, fried foods, sweets, soft drinks)?', ['yes', 'no']))
# fcvc
input_fcvc = str(st.selectbox(
    'How often do you eat vegetables (greens and vegetables) during your main meals?', ['Rarely', 'Sometimes', 'Always']))
input_fcvc_dic = {'Rarely': 1, 'Sometimes': 2, 'Always': 3}
input_fcvc = input_fcvc_dic.get(input_fcvc)

# ncp
input_ncp = str(st.selectbox(
    'How many main meals do you usually eat per day?', [1, 2, 3, '4 ou mais']))
input_ncp_dic = {'1': 1, '2': 2, '3': 3, '4 ou mais': 4}
input_ncp = input_ncp_dic.get(input_ncp)

# caec
input_caec = str(st.selectbox(
    'Do you usually eat snacks between main meals?', ['no', 'Sometimes', 'Frequently', 'Always']))


# smoke
input_smoke = str(st.selectbox(
    'Do you currently smoke?', ['no', 'yes']))


# fcvc
input_ch2o = str(st.selectbox(
    'On average, how much water do you drink per day?', ['Less than 1 liter', 'Between 1 and 2 liters', '2 liters or more']))
input_ch2o_dic = {'Less than 1 liter': 1,
                  'Between 1 and 2 liters': 2, '2 liters or more': 3}
input_ch2o = input_ch2o_dic.get(input_ch2o)


# scc
input_scc = str(st.selectbox(
    'Do you usually monitor or track your daily calorie intake?', ['no', 'yes']))


# faf

input_faf = str(st.selectbox(
    'How often do you practice physical activity per week?', ['I do not practice any physical activity', '1 to 2 times per week', '3 to 4 times per week', '5 times per week or more']))
input_faf_dic = {'I do not practice any physical activity': 0,
                 '1 to 2 times per week': 1,
                 '3 to 4 times per week': 2, '5 times per week or more': 3}
input_faf = input_faf_dic.get(input_faf)


# tue

input_tue = str(st.selectbox(
    'On average, how much time per day do you spend using electronic devices?', ['~0 - 2h', '~3 - 5h', '> 5h',]))
input_tue_dic = {'~0 - 2h': 0,
                 '~3 - 5h': 1,
                 '> 5h': 2}
input_tue = input_tue_dic.get(input_tue)

# calc
input_calc = str(st.selectbox(
    'Do you consume alcoholic beverages? If so, how often?', ['no', 'Sometimes', 'Frequently', 'Always']))

# mtrans
input_mtrans = str(st.selectbox(
    'What is your main mode of transportation in your daily routine?', sorted(dados['mtrans'].unique())))


novo_cliente = [input_gender, input_age, input_height, input_weight, input_family_hist,
                input_favc, input_fcvc, input_ncp, input_caec, input_smoke, input_ch2o, input_scc,
                input_faf, input_tue, input_calc, input_mtrans, np.nan]


def data_split(df, test_size):
    df_treino, df_teste = train_test_split(
        df, test_size=test_size, random_state=7)
    return df_treino.reset_index(drop=True), df_teste.reset_index(drop=True)


df_treino, df_teste = data_split(dados, 0.2)

cliente_predict_df = pd.DataFrame([novo_cliente], columns=df_teste.columns)

teste_novo_cliente = pd.concat(
    [df_teste, cliente_predict_df], ignore_index=True)

# Pipeline


def pipeline_teste(df):
    pipeline = Pipeline([
        ('OneHotEncoder', OneHotEncodingNames()),
        ('OrdinalEncoder', ObesityOrdinalEncoder()),
        ('BinaryEncoder', BinaryLabelEncoder()),
        ('MinMaxScaler', MinMax())])
    df_pipeline = pipeline_teste.fit_transform(df)
    return df_pipeline


teste_novo_cliente = pipeline(teste_novo_cliente)

cliente_pred = teste_novo_cliente.drop(['obesity'], axis=1)


CLASSES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

if st.button("Enviar"):
    model = joblib.load("modelo/rf.joblib")

    # predição (pega o valor único previsto para o novo cliente)
    final_pred = model.predict(cliente_pred)[-1]

    # 1) mostra o rótulo "exato" do dataset
    st.success(f"Your prediction is: **{final_pred}**")

    for c in CLASSES:
        if c == final_pred:
            st.markdown(f"✅ **{c}**")
        else:
            st.markdown(f"• {c}")
