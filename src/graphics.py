import streamlit as st
from src.load_dataset import load_dataset

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

def graphics_app(prev_vars=None):
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

    with st.expander('Основные настройки графика'):
        allowed_columns = st.multiselect('Выберите колонки для построения графика', dataset_columns,
                                         default=dataset_columns)

        pairplot_hue_columns = allowed_columns
        for elem in pairplot_hue_columns:
            if len(dataset[elem].unique()) > 100:
                pairplot_hue_columns.remove(elem)

        pairplot_hue = st.selectbox('Столбец для кодирования цета',
                                    [None] + pairplot_hue_columns)

    with st.expander('Настройки дизайна графика'):
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

    with st.expander('Основные настройки графика'):
        distplot_kind = st.selectbox('Выберите тип графика', ['hist', 'kde', 'ecdf'])

        distplot_column = st.selectbox('Выберите столбец для построения распределения',
                                       dataset_columns)
    st.write("\n\n")

    with st.expander('Настройки дизайна графика'):
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

    with st.expander('Основные настройки графика'):
        values_column = st.selectbox('Выберите столбец значений',
                                     dataset_columns)

        names_column = st.selectbox('Выберите столбец, по которому будет построено распределение',
                                    dataset_columns)

        pieplot_hover_data = st.multiselect('Выберите столбцы с дополнительной информацией',
                                            dataset_columns)
    st.write('\n\n')

    with st.expander('Настройки дизайна графика'):
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

    def display_heatmap(df, columns, annot):
        fig, ax = plt.subplots()
        sns.heatmap(df[columns].corr(), ax=ax, annot=annot, linewidths=.5, square=True, fmt='.2f')
        st.write(fig)

    with st.expander('Основные настройки графика'):
        heatmap_annotation = st.selectbox('Нужно ли выводить цифры корреляции', [True, False])

        heatmap_columns = st.multiselect('Выберите числовые столбцы для графика',
                                         numeric_columns, default=numeric_columns)
    st.write('\n\n')

    st.write('\n\n')
    if st.button('Построить график', key='heatmap_button'):
        try:
            display_heatmap(dataset, heatmap_columns, heatmap_annotation)
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