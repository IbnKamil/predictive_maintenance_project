import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Удаление ненужных столбцов
        data = data.drop(columns=['Product ID','UDI', 'TWF', 'HDF', 'PWF', 'OSF','RNF'])

        # Предобработка данных
        data = data.dropna()  # удаление пропусков

        # Преобразование категориальной переменной Product ID в числовую
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Определяем X и y
        if 'Machine failure' not in data.columns:
            st.error('В датасете должна быть колонка с названием Machine failure')
            return
        X = data.drop('Machine failure', axis=1)
        y = data['Machine failure']

        # Разделение на тренировочные и тестовые
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Визуализация результатов
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_rep)

        # ROC-кривая и AUC
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        ax2.plot([0, 1], [0, 1], linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        st.pyplot(fig2)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            type = st.selectbox("Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature [K]")
            process_temp = st.number_input("Process temperature [K]")
            rotational_speed = st.number_input("Rotational speed [rpm]")
            torque = st.number_input("Torque [Nm]")
            tool_wear = st.number_input("Tool wear [min]")
            submitted = st.form_submit_button("Предсказать")

        if submitted:
            # Преобразование введенных данных
            input_df = pd.DataFrame({
                "Type": [LabelEncoder().fit_transform([type])[0]],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotational_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear]
            })



            # Предсказание
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)[:, 1]

            st.write(f"Предсказание (0 - без отказа, 1 - отказ): {prediction[0]}")
            st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")