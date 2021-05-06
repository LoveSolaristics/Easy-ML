import streamlit as st
from multiapp import MultiApp

import pandas as pd
import numpy as np
import io

import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
import seaborn as sns

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from itertools import product
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Easy ML", page_icon='💙',
)

app = MultiApp()


# TODO: добавить оглавления из якорных ссылок на каждую из страниц

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


def home_app():
    dataset = load_dataset()
    st.markdown("""
    # README.md
    
    Это проект, позволяющий вам быстро оценить данные из датасета.
    Также здесь представлена базовая информация об инструментах 
    визуализации данных, а также построении простейших
    моделей на их основе. 
     
    Здесь вы можете:
    - Выбрать и посмотреть на конкретные строки датасета;
    - Построить различные графики зависимостей фичей из датасета;
    - Обучить простейшие модели классификации или регрессии на ваших данных.
    
    В меню слева можно загрузить свой собственный датасет и 
    выбрать страничку дял отображения.
    """)

    st.image('images/main_img.png')


def data_app():
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

    with st.beta_expander('Ограничения для колонок'):
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


def graphics_app():
    dataset = load_dataset()

    PALETTE_LIST_SEABORN = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
                            'BuGn_r',
                            'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu',
                            'GnBu_r',
                            'Greens', 'Greens_r',
                            'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn',
                            'PRGn_r',
                            'Paired', 'Paired_r',
                            'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
                            'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
                            'Purples',
                            'Purples_r',
                            'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
                            'RdYlBu_r',
                            'RdYlGn',
                            'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3',
                            'Set3_r',
                            'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                            'YlGnBu_r',
                            'YlGn_r',
                            'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                            'autumn',
                            'autumn_r', 'binary', 'binary_r',
                            'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r',
                            'cool',
                            'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
                            'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
                            'gist_earth_r',
                            'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar',
                            'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern',
                            'gist_stern_r',
                            'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
                            'gnuplot_r',
                            'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r',
                            'inferno', 'inferno_r', 'magma', 'magma_r', 'mako', 'mako_r',
                            'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r',
                            'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
                            'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
                            'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b',
                            'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight',
                            'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',
                            'viridis_r',
                            'vlag', 'vlag_r', 'winter', 'winter_r']
    dataset_columns = list(dataset.columns)
    numeric_columns = dataset.select_dtypes(include=np.number).columns.tolist()

    st.title('Построение графиков')

    st.markdown("""
    Содержание:\n
    1. <a href='#part2.1'>Парные зависимости</a>
    2. <a href='#part2.2'>График распределения</a>
    3. <a href='#part2.3'>Круговая диаграмма</a>
    4. <a href='#part2.4'>Heatmap</a>
    5. <a href='#part2.5'>Pandas Profiling</a>
    6. <a href='#part2.6'>Полезные ссылочки</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Для построения графиков в Python существует целый ряд библиотек.
    Помимо базового `matplotlib.pyplot` и встроенных возможностей `pandas`, рекомендую обратить 
    внимание на [seaborn](seaborn.pydata.org/)  и [plotnine]
    (plotnine.readthedocs.io/en/stable/). 
    Они используются для построения неинтерактивных графиков.
    
    Также при помощи таких библиотек, 
    как [bokeh](bokeh.org/), 
    [altair](altair-viz.github.io/gallery/index.html#interactive-charts) и 
    [plotly](plotly.com/python/) 
    есть возможность создать не только интерактивные графики, но и анимации!
    
    [Seaborn](seaborn.pydata.org/) 
    содержит наиболее адекватные дефолтные настройки оформления графиков. 
    Если просто добавить в код `import seaborn`, то картинки станут гораздо симпатичнее. 
    Поэтому используем эту библиотеку для визуализации базовых зависимостей в данных.
    """)

    st.markdown("""
    ## <a id='part2.1'>Парные зависимости</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Даный вид графика показывает отношения между всеми парами переменных, которые
    присутствуют в датасете. Также дополнительные парметры pairplot позволяют
    раскрасить графики в соответствии с какой-либо из переменных.
    
    Чтобы построить pairplot нужно использовать функцию `seaborn.pairplot`. 
    Документацию для этой функции можно посмотреть 
    [здесь](seaborn.pydata.org/generated/seaborn.pairplot.html).
    """)

    def display_pairplot(df, columns, hue, palette):
        pairplot = sns.pairplot(df[columns], hue=hue, palette=palette)
        st.pyplot(pairplot)

    with st.beta_expander('Основные настройки графика'):
        allowed_columns = st.multiselect('Выберите колонки для построения графика', dataset_columns,
                                         default=dataset_columns)

        pairplot_hue_columns = allowed_columns
        for elem in pairplot_hue_columns:
            if len(dataset[elem].unique()) > 100:
                pairplot_hue_columns.remove(elem)

        pairplot_hue = st.selectbox('Столбец для кодирования цета',
                                    [None] + pairplot_hue_columns)

    with st.beta_expander('Настройки дизайна графика'):
        pairplot_palette = st.selectbox('Выберите палитру для графика',
                                        PALETTE_LIST_SEABORN, key='pairplot_palette')

    st.write("\n\n")
    if st.button('Построить график', key='pairplot_button'):
        try:
            display_pairplot(dataset, allowed_columns, pairplot_hue, pairplot_palette)
        except Exception:
            st.info("""
            При построении графика произошла ошибка.
            Попробуйте ещё раз или измените параметры отображения.
            """)

    st.markdown("""
    ## <a id='part2.2'>График распределения </a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Распределение значений конкретной фичи из датасета можно визуализировать
    при помощи гистограммы и графика плотности распределения. 
    
    Тот и другой вид диаграмм может быть построен при помощи единой функции
    `seaborn.distplot`. Передевая в неё дополнительный аргумент `kind` можно
    указать, какой конкретно вид диаграммы ввы хотите построить. Прочие параметры
    можно посмотреть 
    [тут](seaborn.pydata.org/generated/seaborn.displot.html#seaborn.displot).
    """)

    def display_distplot(df, column, kind, palette='Blues'):
        sns.set_palette(palette)
        fig = sns.displot(data=df, x=column, kind=kind)
        st.pyplot(fig)

    with st.beta_expander('Основные настройки графика'):
        distplot_kind = st.selectbox('Выберите тип графика', ['hist', 'kde', 'ecdf'])

        distplot_column = st.selectbox('Выберите столбец для построения распределения',
                                       dataset_columns)
    st.write("\n\n")

    with st.beta_expander('Настройки дизайна графика'):
        distplot_palette = st.selectbox('Выберите палитру для графика',
                                        PALETTE_LIST_SEABORN)
    st.write("\n\n")

    if st.button('Построить график', key='distplot_button'):
        try:
            display_distplot(dataset, distplot_column, distplot_kind, distplot_palette)
        except Exception:
            st.info("""
            При построении графика произошла ошибка.
            Попробуйте ещё раз или измените параметры отображения.
            """)

    st.markdown("""
    ## <a id='part2.3'>Круговая диаграмма</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Круговая диаграмма служит ровно для тех же целей, что и гистограмма, но 
    с её помощью можно визуализировать именно процентное распределение данных.
    
    Ради разнообразия, построим её с помощью библиотеки [plotly](plotly.com/python/).
    Нам понадобится функция `plotly.express.pie`. Графики, построенные с помощью этой библиотеки
    являются интерктивными, так что помимо основной информации можно добавить дополнительную, 
    которая будет отображаться при наведении на части диаграммы. Больше примеров использования
    можно посмотреть по [ссылке](plotly.com/python/pie-charts).
    """)

    def display_pieplot(df, values, names, hover_data=None, template=None):
        if values == names:
            st.info('Выберите различные столбцы. Иначе круговая диаграмма будет неинфромативной')
        else:
            fig = px.pie(df, values=values, names=names, hover_data=hover_data, template=template)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)

    with st.beta_expander('Основные настройки графика'):
        values_column = st.selectbox('Выберите столбец значений',
                                     dataset_columns)

        names_column = st.selectbox('Выберите столбец, по которому будет построено распределение',
                                    dataset_columns)

        pieplot_hover_data = st.multiselect('Выберите столбцы с дополнительной информацией',
                                            dataset_columns)
    st.write('\n\n')

    with st.beta_expander('Настройки дизайна графика'):
        pieplot_template = st.selectbox('Выберите тему для графика',
                                        [None] + list(pio.templates))
    st.write('\n\n')

    if st.button('Построить график', key='pieplot_button'):
        try:
            display_pieplot(dataset, values_column, names_column, pieplot_hover_data,
                            pieplot_template)
        except Exception:
            st.info("""
            При построении графика произошла ошибка.
            Попробуйте ещё раз или измените параметры отображения.
            """)

    st.markdown("""
    ## <a id='part2.4'>Heatmap</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Для того, чтобы понять, насколько в числовом эквиваленте скоррелированы признаки
    можно использовать тепловую карту. Перед этим нужно получить матрицу корреляций
    с помощью метода `DataFrame.corr`, после чего передать получившуюся матрицу в 
    функцию `seaborn.heatmap`. Последняя может не только цвветом показать корреляции,
    но и отобразить числовые значения. Подробности можно почитать в 
    [документации](seaborn.pydata.org/generated/seaborn.heatmap.html).
    """)

    def display_heatmap(df, columns, annot, palette='Blues'):
        fig, ax = plt.subplots()
        sns.heatmap(df[columns].corr(), ax=ax, annot=annot, linewidths=.5, square=True, fmt='.2f',
                    palette=palette)
        st.write(fig)

    with st.beta_expander('Основные настройки графика'):
        heatmap_annotation = st.selectbox('Нужно ли выводить цифры корреляции', [True, False])

        heatmap_columns = st.multiselect('Выберите числовые столбцы для графика',
                                         numeric_columns, default=numeric_columns)
    st.write('\n\n')

    with st.beta_expander('Настройки дизайна графика'):
        heatmap_palette = st.selectbox('Выберите тему для графика',
                                       PALETTE_LIST_SEABORN)

    st.write('\n\n')
    if st.button('Построить график', key='heatmap_button'):
        try:
            display_heatmap(dataset, heatmap_columns, heatmap_annotation, heatmap_palette)
        except Exception:
            st.info("""
            При построении графика произошла ошибка.
            Попробуйте ещё раз или измените параметры отображения.
            """)

    st.markdown("""
    ## <a id='part2.5'>Pandas Profiling</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Профилирование – процесс, который помогает понять наши данные, а [Pandas Profiling](
    towardsdatascience.com/10-simple-hacks-to-speed-up-your-data-analysis-in-python-ec18c6396e6b) 
    – Python библиотека, которая делает это. Простой и быстрый способ выполнить предварительный 
    анализ данных Python Pandas DataFrame. 
    
    Функции pandas df.describe() и df.info(), как правило, становятся первым шагом в 
    автоматизации проектирования электронных устройств. Но это даёт лишь базовое представление о 
    данных и мало помогает при больших наборах. Зато функция Pandas Profiling отображает много 
    информации с помощью одной строки кода и в интерактивном HTML-отчёте.

    Для набора данных пакет Pandas Profiling вычисляет следующую статистику:
    """)

    st.image('images/profiling.png',
             caption='Вычисление статистики в пакете Pandas Profiling')

    st.markdown("""Всё, что для этого нужно - это пара строк кода.""")

    st.code("""
    # импорт необходимых пакетов
    import pandas as pd
    import pandas_profiling
    
    pandas_profiling.ProfileReport(your_dataset)
    """)

    if st.button('Построить статистики', key='profile_button'):
        profile = ProfileReport(dataset)
        st_profile_report(profile)

    st.markdown("""
    ## <a id='part2.6'>Полезные ссылочки</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    1. [The Python Graph Gallery](www.python-graph-gallery.com/).
        Здесь можно посмотреть на огромное количество видов графиков,
        которые можно построить с использованием Python.
    2. [Сравнение библиотек визуализации](pythonplot.com/).
        Тут есть возможность увидеть, чем отличаются одни и те же
        виды графиков, построенные разными библиотеками. Это не только 
        сраввнение внешнего вида, но и количества кода, необходимого
        для реализации базового функционала.
    3. [Seaborn documentation](seaborn.pydata.org/index.html).
        Документация к библиотеке seaborn, содержащая большое количество примеров.
    4. [Color Guide to Seaborn Palettes]
    (medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f).
        Небольшая статья с Medium, в которой автор не только перечисляет все возможные
        палитры цветов для использования в seaborn, но еще и строит графики
        с их использованием, чтобы можно было сразу решить, что вы хотите
        использовать в собственных визуализациях.
    5. [Plotly Open Source Python Library](plotly.com/python/).
        Документация к библиотеке Plotly. Несколько запутанная, но зато снабженная 
        огромным количеством живых примеров.
        
    6. [Pandas Profiling](github.com/pandas-profiling/pandas-profiling).
        Ссылка на гитхаб проекта. Там можно посмотреть, как эта популярная 
        библиотека реализована изнутри.
    """)


def model_app():
    dataset = load_dataset()
    dataset_columns = list(dataset.columns)
    numeric_columns = dataset.select_dtypes(include=np.number).columns.tolist()

    st.markdown("""
    # Обучение моделей
    
    Многие задачи, решаемые с помощью ML, относятся к одной из двух следующих категорий:
    
    1. Задача регрессии – прогноз на основе выборки объектов с различными признаками. 
    На выходе должно получиться вещественное число (2, 35, 76.454 и др.), 
    к примеру цена квартиры, стоимость ценной бумаги по прошествии полугода, 
    ожидаемый доход магазина на следующий месяц, качество вина при слепом тестировании.
    2. Задача классификации – получение категориального ответа на основе набора признаков. 
    Имеет конечное количество ответов (как правило, в формате «да» или «нет»): 
    есть ли на фотографии кот, является ли изображение человеческим лицом, болен ли пациент раком.
    
    Про другие можно прочесть, например, в [статье](habr.com/ru/post/448892/) на Хабре.
    Инструменты, представленные на этой странице позволяют быстро проверить, как реализуют
    себя те или иные алгоритмы на данных, представленных в датасете. То есть вы можете сказать, 
    значения в каком столбце нужно предсказать, данные разобъются на тренировочную и 
    тестовую выборки, модель обучится, предскажет значения на тестовой выборке, а на выходе
    можно будет узнать успешность предсказания, используя нужную метрику.
    
    Выберите задачу, которая стоит перед вами: классификация или регрессия.
    """)

    task = st.selectbox('Задача', [' ', 'Классификация', 'Регрессия'])

    if task == 'Классификация':
        classification_target = st.selectbox(
            'Выберите колонку, значения которой будем предсказывать',
            [None] + dataset_columns)

        if classification_target is None:
            st.info('Выберите target. Для стандартного датасета следует выбрать столбец Survived.')
        else:
            with st.beta_expander('Дополнительные настройки классификации'):
                classification_test_size = st.slider('Размер тестовой выборки от всего '
                                                     'объема датасета', min_value=0.1, max_value=0.5,
                                                     step=0.05, value=0.3)

                features = numeric_columns
                if classification_target in features:
                    features.remove(classification_target)
                classification_features = st.multiselect('Признаки, используемые для классификации',
                                                         features, default=features)

                classification_random_state = st.selectbox('Хотите, чтобы при перезапуске программа '
                                                           'выдавала те же значения?', [False, True])
                if classification_random_state:
                    classification_random_state = 17
                else:
                    classification_random_state = None

                models = {
                    "Nearest Neighbors": KNeighborsClassifier(3),
                    "Linear SVM": SVC(kernel="linear", C=0.025, probability=True),
                    "RBF SVM": SVC(gamma=2, C=1, probability=True),
                    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
                    "Decision Tree": DecisionTreeClassifier(max_depth=5),
                    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10,
                                                            max_features=1),
                    "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
                    "AdaBoost": AdaBoostClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "QDA": QuadraticDiscriminantAnalysis()}

                classification_model_name = st.selectbox('Выберите модель классификации',
                                                         list(models.keys()))

            if st.button('Обучить модель', key='model_button'):
                df = dataset.dropna()

                X = df[classification_features]
                y = df[classification_target]

                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=
                                                                    classification_test_size,
                                                                    random_state=
                                                                    classification_random_state)

                model = models[classification_model_name]
                model.fit(X_train, y_train)

                def plot_confusion_matrix(cm, classes, normalize=False, title='Матрица ошибок'):

                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.title(title)
                    plt.colorbar()
                    tick_marks = np.arange(len(classes))
                    plt.xticks(tick_marks, classes, rotation=45)
                    plt.yticks(tick_marks, classes)

                    if normalize:
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        st.info("Построена ормализованная матрица ошибок")
                    else:
                        st.info('Построена матрица ошибок без нормализации')

                    thresh = cm.max() / 2.
                    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                        plt.text(j, i, cm[i, j],
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

                    plt.tight_layout()
                    plt.ylabel('Истинные метки класса')
                    plt.xlabel('Предсказанные метки класса')

                font = {'size': 15}
                plt.rc('font', **font)
                plt.figure(figsize=(10, 8))

                cnf_matrix = confusion_matrix(y_test, model.predict(X_test))

                st.markdown("""
                ## Матрица ошибок
                
                Перед переходом к самим метрикам необходимо ввести важную концепцию для описания 
                этих метрик в терминах ошибок классификации — confusion matrix (матрица ошибок).
                Допустим, что у нас есть два класса и алгоритм, предсказывающий принадлежность 
                каждого объекта одному из классов, тогда матрица ошибок классификации будет 
                выглядеть следующим образом:
                
                <table style="margin: 10px auto">
                  <tr>
                    <td></td>
                    <td>y = 0</td>
                    <td>y = 1</td>
                  </tr>
                  <tr>
                    <td>y' = 1</td>
                    <td>True Positive (TP)</td>
                    <td>False Positive (FP)</td>
                  </tr>
                  <tr>
                    <td>y' = 0</td>
                    <td>False Negative (FN)</td>
                    <td>True Negative (TN)</td>
                  </tr>
                </table>
                
                Здесь y' — это ответ алгоритма на объекте, а y — истинная метка класса 
                на этом объекте. Таким образом, ошибки классификации бывают двух видов: 
                False Negative (FN) и False Positive (FP).
                """, unsafe_allow_html=True)

                plot_confusion_matrix(cnf_matrix, classes=['Non-churned', 'Churned'],
                                      title='Матрица ошибок')

                st.pyplot(plt)

                st.markdown("""
                ## Precision, recall и F-мера
                
                Для оценки качества работы алгоритма на каждом из классов по отдельности 
                введем метрики $precision$ (точность) и $recall$ (полнота).
            
                """)
                st.latex(r'precision = \frac{TP}{TP + FP}')
                st.latex(r'recall = \frac{TP}{TP + FN}')

                st.markdown("""
                $Precision$ можно интерпретировать как долю объектов, названных классификатором 
                положительными и при этом действительно являющимися положительными, 
                а $recall$ показывает, какую долю объектов положительного класса из всех 
                объектов положительного класса нашел алгоритм.
                
                Именно введение $precision$ не позволяет нам записывать все объекты в один класс, 
                так как в этом случае мы получаем рост уровня False Positive. 
                $Recall$ демонстрирует способность алгоритма обнаруживать данный класс вообще, 
                а $precision$ — способность отличать этот класс от других классов.
                
                Обычно при оптимизации гиперпараметров алгоритма (например, в случае перебора по 
                сетке `GridSearchCV` ) используется одна метрика, улучшение которой мы и ожидаем 
                увидеть на тестовой выборке.
                
                Существует несколько различных способов объединить precision и recall в 
                агрегированный критерий качества. $F-мера$ (в общем случае $F_β$) — среднее 
                гармоническое $precision$ и $recall$:
                """)

                st.latex(r"F_β = (1 + \beta^2) \cdot \frac{precision \cdot recall}"
                         r"{(β^2 \cdot precision) + recall}")

                st.markdown("""$β$ в данном случае определяет вес точности в метрике, и при $β=1$ 
                это среднее гармоническое (с множителем 2, чтобы в случае 
                $precision=1$ и $recall=1$ иметь $F_1=1$). 
                $F-мера$ достигает максимума при полноте и точности, равными единице, 
                и близка к нулю, если один из аргументов близок к нулю.
                В `sklearn` есть удобная функция `metrics.classification_report`, 
                возвращающая $recall$, $precision$ и $F-меру$ для каждого из классов, 
                а также количество экземпляров каждого класса.
                """)

                report = classification_report(y_test, model.predict(X_test))
                st.code('Classification report:\n\n' + report)

                st.markdown("""
                ## ROC-кривая
                
                При конвертации вещественного ответа алгоритма (как правило, вероятности 
                принадлежности к классу, отдельно см. SVM) в бинарную метку, мы должны выбрать 
                какой-либо порог, при котором 0 становится 1. Естественным и близким кажется порог, 
                равный 0.5, но он не всегда оказывается оптимальным, например,
                отсутствии баланса классов.

                Одним из способов оценить модель в целом, не привязываясь к конкретному порогу, 
                является AUC-ROC (или ROC AUC) — площадь (Area Under Curve) под кривой ошибок 
                (Receiver Operating Characteristic curve ). Данная кривая представляет из себя 
                линию от (0,0) до (1,1) в координатах True Positive Rate (TPR) и False Positive 
                Rate (FPR):
                """)

                st.latex(r"TPR=\frac{TP}{TP + FN}")
                st.latex(r"FPR=\frac{FP}{FP + TN}")

                st.write("""TPR нам уже известна, это полнота, а FPR показывает, какую долю из 
                объектов negative класса алгоритм предсказал неверно. В идеальном случае, 
                когда классификатор не делает ошибок (FPR = 0, TPR = 1) мы получим площадь под 
                кривой, равную единице; в противном случае, когда классификатор случайно выдает 
                вероятности классов, AUC-ROC будет стремиться к 0.5, так как классификатор будет 
                выдавать одинаковое количество TP и FP.
                Каждая точка на графике соответствует выбору некоторого порога. Площадь под 
                кривой в данном случае показывает качество алгоритма (больше — лучше), кроме этого, 
                важной является крутизна самой кривой — мы хотим максимизировать TPR, минимизируя 
                FPR, а значит, наша кривая в идеале должна стремиться к точке (0,1).
                """)

                sns.set(font_scale=1.5)
                sns.set_color_codes("muted")

                plt.figure(figsize=(10, 8))

                fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1],
                                                 pos_label=1)

                lw = 2
                plt.plot(fpr, tpr, lw=lw, label='ROC curve')
                plt.plot([0, 1], [0, 1])
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC-кривая')
                st.pyplot(plt)
                # TODO:
                #  разобраться с предупреждением
                #  UndefinedMetricWarning: No positive samples in y_true, true positive value
                #  should be meaningless на Ирисах Фишера

                st.markdown("""
                Критерий AUC-ROC устойчив к несбалансированным классам 
                (спойлер: увы, не всё так однозначно) и может быть интерпретирован как вероятность 
                того, что случайно выбранный positive объект будет проранжирован классификатором 
                выше (будет иметь более высокую вероятность быть positive), чем случайно выбранный 
                negative объект.
                """)

                st.markdown("""
                ## Полезные ссылки
                
                Разумеется здесь рассморены не все возможные метрики в задаче классификации, 
                например, "забыли" про банальный accuracy из-за плохой интерпретируемости
                результатов, а также не рассмотрели, к примеру, logic loss. Но в один текст это
                всё не уместишь, поэтому ниже ссылки для дальнейшего ознакомления:
                
                1. [Classifier comparison](
                scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
                . Примеры классификаторов из библиотеки `sklearn` с красивыми графиками.
                2. [Метрики в задачах машинного обучения]
                (habr.com/ru/company/ods/blog/328372/). Текст о метриках в основном взят
                именно оттуда. 
                3. [Metrics and scoring: quantifying the quality of predictions]
                (scikit-learn.org/stable/modules/model_evaluation.html). Очередная ссылка
                на документацию `sklearn`, но на этот раз на страницу, где подробнее говорится о 
                метриках.
                """)

    elif task == 'Регрессия':
        regression_target = st.selectbox(
            'Выберите колонку, значения которой будем предсказывать',
            [None] + numeric_columns)

        if regression_target is None:
            st.info('Выберите target. Для стандартного датасета следует выбрать столбец Survived.')
        else:
            with st.beta_expander('Дополнительные настройки регрессии'):
                regression_n_splits = st.slider('Кроличество разбиений', min_value=5, max_value=20,
                                                step=1, value=10)

                features = numeric_columns
                if regression_target in features:
                    features.remove(regression_target)
                regression_features = st.multiselect('Признаки, используемые для классификации',
                                                     features, default=features)

                regression_random_state = st.selectbox('Хотите, чтобы при перезапуске программа '
                                                       'брала на train и test те же данные?',
                                                       [False, True])
                if regression_random_state:
                    regression_random_state = 17
                else:
                    regression_random_state = None

                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso(),
                    "ElasticNet": ElasticNet(),
                    "KNeighborsRegressor": KNeighborsRegressor(),
                    "Decision TreeRegressor": DecisionTreeRegressor(),
                    "SVR": SVR()
                }

                regression_model_name = st.selectbox('Выберите модель классификации',
                                                     list(models.keys()))

                metrics = {
                    "Средняя абсолютная ошибка (MAE)": 'neg_mean_absolute_error',
                    "Средняя квадратическая ошибка (MSE)": 'neg_mean_squared_error',
                    "R Squared": 'r2'
                }

            if st.button('Обучить модель', key='model_button'):
                df = dataset.dropna()

                X = df[regression_features]
                y = df[regression_target]

                model = models[regression_model_name]

                kfold = KFold(n_splits=10,
                              random_state=regression_random_state)

                results = pd.DataFrame(columns=['Название метрики', 'Среднее значение',
                                                'Стандартное отклонение'])
                for metric_name, scoring in metrics.items():
                    result = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
                    results = results.append({'Название метрики': metric_name,
                                              'Среднее значение': result.mean(),
                                              'Стандартное отклонение': result.std()},
                                             ignore_index=True)

                st.markdown("""
                В этом разделе будут рассмотрены 3 наиболее распространенных показателя для оценки 
                прогнозов проблем регрессионного машинного обучения:
                
                1. Средняя абсолютная ошибка.
                2. Средняя квадратичная ошибка
                3. $R^2$.
                
                ## Средняя абсолютная ошибка
                Средняя абсолютная ошибка (или MAE) представляет собой сумму абсолютных различий 
                между прогнозами и фактическими значениями. Это дает представление о том, 
                насколько неправильными были прогнозы.

                Мера дает представление о величине ошибки, но не дает представление о
                направлении. То есть при определении скорости машины мы могли ошибиться на 20 км/ч,
                но сказать, двигался ли автомобиль медленне или быстрее предсказанного не смогли бы.
                 
                ## Средняя квадратическая ошибка
                
                Средняя квадратическая ошибка (или MSE) очень похожа на среднюю абсолютную 
                ошибку в том, что она дает общее представление о величине ошибки.

                Взятие квадратного корня из среднеквадратичной ошибки преобразует единицы обратно 
                в исходные единицы выходной переменной и может иметь смысл для описания и 
                представления. Это называется среднеквадратической ошибкой (или RMSE).
                
                ## Метрика $R ^ 2$
                
                Метрика $R ^ 2$ (или R Squared) указывает на достоверность соответствия набора 
                прогнозов фактическим значениям. В статистической литературе эта мера называется 
                коэффициентом детерминации.

                Это значение между 0 и 1 для неподходящего и идеального соответствия соответственно.
                
                ## Оценка работы алгоритма
                
                Данные из датасета были разбиты на несколько частей определенное количество раз с
                помощью `sklearn.model_selection.cross_val_score` и `sklearn.model_selection.Kfold`,
                каждый раз после обучения модели был произведен рассчет трех вышеописанных метрик.
                Средний результат вы можете увидеть в таблице ниже:
                """)

                st.dataframe(results)

                st.markdown("""
                Также для наглядност построим scatterplot, чтобы понять, как результаты работы
                алгоритма отличаются от реальных. В идеальном случае точки должны лежать на одной 
                прямой, направленной под углом 45 градуусов.
                """)

                x, y = df[regression_features], df[regression_target]
                model = models[regression_model_name]
                model.fit(x,y)
                y_pred = model.predict(x)
                plt.scatter(y_pred, y)
                plt.title("Предсказание vs реальность")
                plt.xlabel("Предсказанные значения")
                plt.ylabel("Реальные значения")
                st.pyplot(plt)

                st.markdown("""
                ## Полезные ссылки
                
                1. [Документация о кросс-валидации] ( 
                scikit-learn.org/stable/modules/cross_validation.html). Здесь можно узнать больше 
                об уже упомянутом способе кросс-валидации, а также узнать, какие еще способы 
                существуют, и как их использовать. 
                2. [Статья о метриках в задачах классификации и регрессии]
                (machinelearningmastery.ru/metrics-evaluate-machine-learning-algorithms-python/). 
                Короткая обзорная статья с примерами кода для новичков. 
                3. [MachineLearning.ru](
                machinelearning.ru/wiki/index.php?title=Заглавная_страница). 
                Профессиональный информационно-аналитический ресурс, посвященный
                машинному обучению, распознаванию образов и интеллектуальному анализу данных. 
                4. [Разбор задачи регрессии на примере датасета Boston](
                kaggle.com/shreayan98c/boston-house-price-prediction/data?select=housing.csv)
                """)

    else:
        st.info('Выберите задачу для продолжения')


# Add all your application here

app.add_app("README.md", home_app)
app.add_app("Просмотр датасета через Pandas", data_app)
app.add_app("Построение графиков", graphics_app)
app.add_app("Обучение моделей", model_app)

# The main app
app.run()
