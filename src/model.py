import streamlit as st
from src.load_dataset import load_dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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

def model_app(prev_vars=None):
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
            with st.expander('Дополнительные настройки классификации'):
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
            with st.expander('Дополнительные настройки регрессии'):
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
                model.fit(x, y)
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