import streamlit as st

import pandas as pd


def load_dataset():
    st.sidebar.markdown("""
    # Загрузка данных

    Вы можете загрузить свой собственный датасет, нажав кнопку ниже, или посмотреть, как работает 
    приложение на тестовых данных из датасета 
    [Titanic - Machine Learning from Disaster](www.kaggle.com/c/titanic/).
    """)

    uploaded_file = st.sidebar.file_uploader('Поддерживаются только файлы типа csv. Лимит: 200MB',
                                             type='csv')

    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

        return dataset

    st.sidebar.info("Сейчас используется стандартный датасет")
    return pd.read_csv('example_dataset.csv')
