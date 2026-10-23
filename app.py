
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

# Все пути считаются относительно папки, где лежит app.py
APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    """Модель загружается один раз и переиспользуется."""
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_config():
    with open(APP_DIR / "config.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def get_test_probabilities(_model, X):
    """Вероятности тестовых объектов вычисляются и кэшируются."""
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title="Wine Quality Assistant", layout="wide")
st.title("🍷 Wine Quality Assistant")
st.write(
    "Инструмент для оценки вероятности того, что вино относится "
    "к хорошим, и изучения работы модели."
)

try:
    model = load_model()
    config = load_config()
    test_df = load_test_data()
except Exception as e:
    st.error(f"Не удалось загрузить файлы приложения: {e}")
    st.stop()

if "target" not in test_df.columns:
    st.error("В test.csv отсутствует столбец target.")
    st.stop()

feature_names = [c for c in test_df.columns if c != "target"]
X_test = test_df[feature_names].copy()
y_test = test_df["target"].astype(int)

default_threshold = float(config.get("threshold", 0.5))

# Общий порог используется во всех вкладках
st.sidebar.header("Настройки модели")
threshold = st.sidebar.slider(
    "Порог хорошего вина",
    min_value=0.0,
    max_value=1.0,
    value=float(np.clip(default_threshold, 0.0, 1.0)),
    step=0.01,
    key="threshold",
    help="Вино считается хорошим, если вероятность не меньше порога.",
)
st.sidebar.write(f"Текущий порог: **{threshold:.2f}**")

tab_single, tab_metrics, tab_batch, tab_coeff = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

# ---------------------------------------------------------
# Вкладка 1. Оценка одного вина
# ---------------------------------------------------------
with tab_single:
    st.subheader("Оценить одно вино")
    st.write(
        "Введите характеристики вина. По умолчанию используются "
        "медианы тестовой выборки."
    )

    with st.form("single_wine_form"):
        values = {}

        # Числовые поля; для каждого используем диапазон test.csv
        numeric_features = [
            c for c in feature_names if c != "is_red"
        ]

        cols = st.columns(2)
        for i, col in enumerate(numeric_features):
            series = pd.to_numeric(test_df[col], errors="coerce").dropna()
            minimum = float(series.min())
            maximum = float(series.max())
            median = float(series.median())

            # number_input требует max_value >= min_value
            if minimum == maximum:
                minimum -= 0.01
                maximum += 0.01

            with cols[i % 2]:
                values[col] = st.number_input(
                    col.replace("_", " ").capitalize(),
                    min_value=minimum,
                    max_value=maximum,
                    value=float(np.clip(median, minimum, maximum)),
                    format="%.4f",
                    key=f"single_{col}",
                )

        if "is_red" in feature_names:
            is_red = st.toggle("Красное вино", value=False)
            values["is_red"] = int(is_red)

        submitted = st.form_submit_button("Оценить вино")

    if submitted:
        wine = pd.DataFrame([values], columns=feature_names)
        probability = float(model.predict_proba(wine)[0, 1])
        prediction = int(probability >= threshold)

        st.metric("Вероятность хорошего вина", f"{probability:.1%}")
        if prediction == 1:
            st.success("Модель рекомендует это вино как хорошее.")
        else:
            st.warning("Модель не относит это вино к хорошим при текущем пороге.")

        st.caption(
            f"Порог: {threshold:.2f}. При его изменении решение может измениться."
        )

# ---------------------------------------------------------
# Вкладка 2. Метрики и кривые
# ---------------------------------------------------------
with tab_metrics:
    st.subheader("Качество модели на тестовой выборке")

    p_test = get_test_probabilities(model, X_test)
    y_pred = (p_test >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1", f"{f1:.3f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots()
    ax_cm.imshow(cm)
    ax_cm.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Предсказано 0", "Предсказано 1"],
        yticklabels=["Истинный 0", "Истинный 1"],
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        title="Матрица ошибок",
    )
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # ROC-кривая и текущая точка
    fpr, tpr, _ = roc_curve(y_test, p_test)
    roc_auc = roc_auc_score(y_test, p_test)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", label="Случайный классификатор")
    fp_rate = cm[0, 1] / max(cm[0].sum(), 1)
    tp_rate = cm[1, 1] / max(cm[1].sum(), 1)
    ax_roc.scatter([fp_rate], [tp_rate], color="red", label="Текущий порог")
    ax_roc.set(xlabel="FPR", ylabel="TPR / Recall", title="ROC-кривая")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # PR-кривая и текущая точка
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, p_test)
    pr_auc = average_precision_score(y_test, p_test)

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.3f}")
    ax_pr.scatter(
        [recall], [precision],
        color="red", s=70, label="Текущий порог"
    )
    ax_pr.axhline(
        y=float(y_test.mean()),
        linestyle="--",
        label="Доля хороших вин"
    )
    ax_pr.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall кривая",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax_pr.legend()
    ax_pr.grid(alpha=0.3)
    st.pyplot(fig_pr)
    plt.close(fig_pr)

    st.write(
        "**Как влияет порог?** При уменьшении порога модель обычно "
        "рекомендует больше вин: recall может вырасти, но precision "
        "может снизиться. При увеличении порога рекомендаций становится "
        "меньше; precision может вырасти, но часть хороших вин будет пропущена. "
        "Конкретные изменения зависят от данных."
    )

# ---------------------------------------------------------
# Вкладка 3. Пакетное предсказание
# ---------------------------------------------------------
with tab_batch:
    st.subheader("Загрузить CSV с несколькими винами")
    st.write(
        "CSV должен содержать все признаки модели и не должен содержать "
        "target. Названия столбцов должны совпадать с test.csv."
    )

    uploaded_file = st.file_uploader(
        "Выберите CSV-файл",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing = [c for c in feature_names if c not in batch_df.columns]

            if missing:
                st.error(
                    "В файле не хватает обязательных столбцов: "
                    + ", ".join(missing)
                )
            else:
                batch_X = batch_df[feature_names].copy()

                # Проверяем, что все признаки числовые
                for col in feature_names:
                    batch_X[col] = pd.to_numeric(
                        batch_X[col], errors="coerce"
                    )

                if batch_X.isna().any().any():
                    st.error(
                        "В обязательных столбцах есть пропуски или "
                        "нечисловые значения. Исправьте CSV и загрузите его снова."
                    )
                else:
                    batch_prob = model.predict_proba(batch_X)[:, 1]
                    result = batch_df.copy()
                    result["probability_good"] = batch_prob
                    result["prediction"] = (
                        batch_prob >= threshold
                    ).astype(int)
                    result["decision"] = np.where(
                        result["prediction"] == 1,
                        "Хорошее",
                        "Не отнесено к хорошим",
                    )

                    st.dataframe(result, use_container_width=True)

                    csv_bytes = result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Скачать результаты CSV",
                        data=csv_bytes,
                        file_name="wine_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Не удалось обработать CSV-файл: {e}")

# ---------------------------------------------------------
# Вкладка 4. Коэффициенты логистической регрессии
# ---------------------------------------------------------
with tab_coeff:
    st.subheader("Коэффициенты модели")
    st.write(
        "Положительный коэффициент повышает логарифм шансов класса 1, "
        "отрицательный — понижает, при прочих равных."
    )

    # Достаём LogisticRegression из sklearn- или imblearn-пайплайна
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        lr_candidates = [
            step for step in steps.values()
            if hasattr(step, "coef_")
        ]
        lr = lr_candidates[-1] if lr_candidates else None
    else:
        lr = model if hasattr(model, "coef_") else None

    if lr is None:
        st.error("Не удалось найти обученную логистическую регрессию в пайплайне.")
    else:
        coefficients = pd.DataFrame({
            "Признак": feature_names,
            "Коэффициент": lr.coef_[0],
        }).sort_values("Коэффициент")

        fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
        ax_coef.barh(
            coefficients["Признак"],
            coefficients["Коэффициент"],
        )
        ax_coef.axvline(0, color="black", linewidth=0.8)
        ax_coef.set_xlabel("Коэффициент логистической регрессии")
        ax_coef.set_title("Вклад признаков после масштабирования")
        fig_coef.tight_layout()
        st.pyplot(fig_coef)
        plt.close(fig_coef)

        negative = coefficients.head(3)["Признак"].tolist()
        positive = coefficients.tail(3).sort_values(
            "Коэффициент", ascending=False
        )["Признак"].tolist()

        st.write("**Сильнее всего голосуют за хорошее вино:** " +
                 ", ".join(positive))
        st.write("**Сильнее всего голосуют против:** " +
                 ", ".join(negative))

        st.caption(
            "Коэффициенты показывают связь при фиксированных остальных "
            "признаках, а не доказывают причинно-следственную зависимость."
        )
