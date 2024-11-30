import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.stats import pearsonr
import streamlit as st
import matplotlib.pyplot as plt


# --- Функции программы ---
def load_data(file_path):
    """Загрузка данных из текстового файла"""
    return pd.read_excel(file_path)


def preprocess_data(data):
    """Обработка данных: проверка на пропуски"""
    data = data.dropna()
    return data


def build_model(X, y):
    """Оценка линейной многофакторной модели (ЛМФМ)"""
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model


def evaluate_model(model, X_test, y_test):
    """Оценка модели: R^2, RMSE"""
    y_pred = model.predict(sm.add_constant(X_test))
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return r2, rmse, y_pred


def get_statistical_info(X, y):
    """Получение статистической информации о факторах"""
    stats = []
    for col in X.columns:
        corr, pval = pearsonr(X[col], y)
        stats.append({"Factor": col, "Correlation with y": corr, "p-value:": pval})
    corr = X.corr()
    corr.columns = X.columns.copy()
    corr.index = X.columns.copy()
    return pd.DataFrame(stats), corr


def plot_results(y_train, y_train_pred, y_test, y_test_pred):
    """Визуализация результатов"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index, y_train, label="Actual (Train)", color="blue")
    ax.plot(
        y_train.index,
        y_train_pred,
        label="Predicted (Train)",
        linestyle="--",
        color="orange",
    )
    ax.plot(y_test.index, y_test, label="Actual (Test)", color="green")
    ax.plot(
        y_test.index, y_test_pred, label="Predicted (Test)", linestyle="--", color="red"
    )
    ax.set_title("Model Predictions vs Actual")
    ax.legend()
    return fig


# --- Интерфейс Streamlit ---
st.title("Линейная Многофакторная Модель (ЛМФМ) с Итеративным Отбором")

# Шаг 1: Загрузка данных
st.sidebar.header("Шаг 1: Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите файл данных", type=["xlsx"])

if uploaded_file:
    data = load_data(uploaded_file)
    st.write("### Просмотр данных")
    st.dataframe(data)

    # Шаг 2: Выбор параметров
    st.sidebar.header("Шаг 2: Параметры")
    target_column = st.sidebar.selectbox("Выберите колонку отклика (y)", data.columns)
    st.session_state["feature_columns"] = st.sidebar.multiselect(
        "Выберите колонки факторов (X)",
        [col for col in data.columns if col != target_column],
    )
    if data is not None:
        test_size = st.sidebar.slider("Размер тестовой выборки", 0, data.shape[0], 5, 5)

    if target_column and st.session_state["feature_columns"]:
        data = data.sort_values("date", ascending=True)
        X = data[st.session_state["feature_columns"]]
        y = data[target_column]

        # Предобработка данных
        X_test = X.loc[X.shape[0] - test_size :, :]
        X_train = X.loc[: X.shape[0] - test_size, :]
        y_test = y.loc[X.shape[0] - test_size :]
        y_train = y.loc[: X.shape[0] - test_size]

        # Итеративный отбор факторов
        st.markdown("# Итеративный отбор факторов")
        exclude_columns = []
        st.session_state["remaining_columns"] = st.session_state[
            "feature_columns"
        ].copy()

        # Вывод статистики
        X_iter = X_train[st.session_state["remaining_columns"]]
        stats_df, corr = get_statistical_info(X_iter, y_train)
        st.write("Корреляция с таргетом")
        st.dataframe(stats_df)
        st.write("Корреляция между переменными")
        st.dataframe(corr)

        # Выбор факторов для исключения
        exclude_column = st.multiselect(
            "Выберите фактор для исключения:", st.session_state["remaining_columns"]
        )

        st.session_state["remaining_columns"] = list(
            set(st.session_state["feature_columns"]).difference(exclude_column)
        )

        # Выбор лагов
        lags = []
        for column in st.session_state["remaining_columns"]:
            lags.append(
                st.selectbox(f"Выберите лаг для переменной {column}", [0, 1, 2, 3], 0)
            )

        # создание переменных для лагов
        for i, lag in enumerate(lags):
            if lag > 0:
                for sub_lag in range(1, lag + 1):
                    X_train[
                        f"{ st.session_state["remaining_columns"][i]}_lag_{sub_lag}"
                    ] = X_train[f"{ st.session_state["remaining_columns"][i]}"].diff(
                        sub_lag
                    )
                    X_test[
                        f"{ st.session_state["remaining_columns"][i]}_lag_{sub_lag}"
                    ] = X_test[f"{ st.session_state["remaining_columns"][i]}"].diff(
                        sub_lag
                    )
                    st.session_state["remaining_columns"].append(
                        f"{ st.session_state["remaining_columns"][i]}_lag_{sub_lag}"
                    )

        max_lag = max(lags)
        # Финальный набор факторов
        st.write("### Финальный набор факторов")
        st.write(st.session_state["remaining_columns"])

        # Построение модели
        X_train_final = X_train.loc[max_lag:, st.session_state["remaining_columns"]]
        X_test_final = X_test.loc[
            X.shape[0] - test_size + max_lag :, st.session_state["remaining_columns"]
        ]
        y_train_final = y_train.loc[max_lag:]
        y_test_final = y_test.loc[X.shape[0] - test_size + max_lag :]
        model = build_model(X_train_final, y_train_final)
        st.write("### Итоговая модель")
        st.code(model.summary())
        # st.text(model.summary())

        # Оценка модели
        r2, rmse, y_test_pred = evaluate_model(model, X_test_final, y_test_final)
        y_train_pred = model.predict(sm.add_constant(X_train_final))

        st.write("### Метрики модели")
        st.metric("R^2", f"{r2:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")

        # Визуализация
        st.write("### График результатов")
        fig = plot_results(y_train_final, y_train_pred, y_test_final, y_test_pred)
        st.pyplot(fig)

        # Прогнозирование
        st.write("### Прогноз на основе новых значений факторов")
        st.write("Введите значения для факторов:")
        user_input = {}
        for col in st.session_state["remaining_columns"]:
            user_input[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Получить прогноз"):
            user_data = pd.DataFrame([user_input])
            user_data["const"] = 1
            user_data = user_data[["const"] + list(user_data.columns)[:-1]]
            prediction = model.predict(user_data)[0]
            st.write(f"**Прогнозируемое значение отклика (y): {prediction:.2f}**")
