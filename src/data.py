import streamlit as st
from src.load_dataset import load_dataset

import io
import numpy as np

def data_app(prev_vars=None):
    dataset = load_dataset()
    st.title('Просмотр датасета через Pandas')

    st.markdown("""
    Содержание:\n
    1. <a href='#part1.1'>Сэмпл данных</a>
    2. <a href='#part1.2'>Информация о датасете</a>
    3. <a href='#part1.3'>Таблица с отфильтрованными значениями</a>
    4. <a href='#part1.4'>Полезные ссылочки</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Прежде чем пытаться визуализировать данных, используя графики, и строить какие-либо модели
    необходимо просто напросто понять, что за данные входят в датасет.

    Для этого стоит взглянуть на конкретные строки таблицы с данными, чтобы понимать,
    в каком форме представлены данные, что можно распарсить, как можно получить
    дополнительные признаки. Всё это легко осуществить с помощью библиотеки `pandas`.
    """)

    st.markdown("""
    ## <a id='part1.1'>Сэмпл данных</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Прежде всего можно взять небольшую произвольную часть данных и посмотреть, как она выглядит.
    Для этого используется метод `pandas.DataFrame.sample`
    """)

    sample_len = st.slider('Величина сэмпла', min_value=5, max_value=10, value=7)

    dataset_sample = dataset.sample(sample_len)
    st.dataframe(dataset_sample)

    if st.button('Обновить сэмпл'):
        dataset_sample = dataset.sample(sample_len)

    st.markdown("""
    ## <a id='part1.2'>Информация о датасете</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Также бывает полезно взглянуть на то, какими типами представлены данные. 
    Для этого используется метод `pandas.DataFrame.info`. 

    Он выводит три столбца: названия колонок, количество ненулевых значений и их тип.
    """)

    buffer = io.StringIO()
    dataset.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown("""
    ## <a id='part1.3'>Таблица с отфильтрованными значениями</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    С помощью специального синтаксиса, предоставляемого библиотекой `pandas` для класса `DataFrame`
    можно с легкостью выбрать строки, которые соответствуют тем или иным логическим значениям.
    Это круче, чем Excel-таблицы и SQL-запросы.

    Возьмем все столбцы с числами (`pandas.DataFrame._get_numeric_data` или более безопасно 
    `pandas.DataFrame.select_dtypes(include=np.number).columns.tolist()`), 
    сделаем для них слайдеры
    и отфильтруем строки, используя значения, введенные через них.
    """)

    numeric_columns = dataset.select_dtypes(include=np.number).columns.tolist()

    sliders = {}

    dataset_columns = list(dataset.columns)
    allowed_columns = st.multiselect('Выберите колонки для отображения', dataset_columns,
                                     default=dataset_columns)

    with st.expander('Ограничения для колонок'):
        for column in numeric_columns:
            sliders[column] = st.slider(column, min(dataset[column]), max(dataset[column]),
                                        (min(dataset[column]), max(dataset[column])))

    filtered_data = dataset.copy()
    for column in numeric_columns:
        filtered_data = filtered_data[(filtered_data[column] >= sliders[column][0]) &
                                      (filtered_data[column] <= sliders[column][1])]

    st.dataframe(filtered_data[allowed_columns])

    st.markdown("""
    ## <a id='part1.4'>Полезные ссылочки</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    И под конец поделюсь ссылками на ресурсы, которые так или иначе были использованы 
    в процессе создания этой странички.

    1. [Pandas User Guide](pandas.pydata.org/docs/user_guide/index.html#user-guide). 
        Очевидно, что неотъемлемой частью использования библиотеки является
        заглядывание в её документацию. 
    2. [Data Analysis with pandas - 30 Days Of Python](youtu.be/g_rfQQC2BjA). Канал 
        хорошего специалиста по Python, у которго есть замечательные курсы на Udemy и информативные
        ролики на Youtube.
    3. [HSE: Наука о данных](math-info.hse.ru/2020-21/Наука_о_данных). Понятный курс для 
        непрограммистов, которым по долгу службы приходится использовать python для анализа данных.
    """)