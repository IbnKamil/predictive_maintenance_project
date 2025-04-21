import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier



def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # -------------------------------------------------------------------------------------
    # -------------------------------ЗАГРУЗКА ДАННЫХ---------------------------------------
    # -------------------------------------------------------------------------------------

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # -------------------------------------------------------------------------------------
        # -------------------------------ПРЕДОБРАБОТКА ДАННЫХ----------------------------------
        # -------------------------------------------------------------------------------------

        # Удаление ненужных столбцов
        data = data.drop(columns=['Product ID', 'UDI', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

        # Преобразование категориальной переменной Type в числовую
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Удаление пропусков
        data = data.dropna()

        # Чтобы программа не выдавало ошибок при загрузке не соответствующего файла
        if 'Machine failure' not in data.columns:
            st.error('В датасете должна быть колонка с названием Machine failure')
            return

        # Определяем X и y
        X = data.drop('Machine failure', axis=1)
        y = data['Machine failure']

        # -------------------------------------------------------------------------------------
        # ------------------РАЗДЕЛЕНИЕ НА ТРЕНИРОВОЧНЫЕ И ТЕСТОВЫЕ ДАННЫЕ----------------------
        # -------------------------------------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабирование: стандартизация данных
        standard = StandardScaler()
        X_train_st = standard.fit_transform(X_train)
        X_test_st = standard.transform(X_test)

        # -------------------------------------------------------------------------------------
        # -------------------------------ЛИНЕЙНАЯ РЕГРЕССИЯ------------------------------------
        # -------------------------------------------------------------------------------------

        # Обучение: 1. Линейной регрессии
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_st, y_train)
        y_pred = model.predict(X_test_st)

        # Оценка модели Линейной регрессии
        accuracy_y = accuracy_score(y_test, y_pred)
        conf_matrix_y = confusion_matrix(y_test, y_pred)
        classification_rep_y = classification_report(y_test, y_pred)

        # Визуализация результатов
        st.header("Результаты обучения модели Линейной регрессии")
        st.write(f"Accuracy: {accuracy_y:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_y, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_rep_y)

        # ROC-кривая и AUC
        y_proba = model.predict_proba(X_test_st)[:, 1]
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

        # -------------------------------------------------------------------------------------
        # ---------------------------------СЛУЧАЙНЫЙ ЛЕС---------------------------------------
        # -------------------------------------------------------------------------------------

        # Обучение: 2. Случайного леса (классификация)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_st, y_train)
        pred_rf = rf.predict(X_test_st)

        # Оценка модели Случайного леса
        accuracy_rf = accuracy_score(y_test, pred_rf)
        conf_matrix_rf = confusion_matrix(y_test, pred_rf)
        classification_rep_rf = classification_report(y_test, pred_rf)

        # Визуализация результатов
        st.header("Результаты обучения модели Случайного леса")
        st.write(f"Accuracy: {accuracy_rf:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_rep_rf)

        # ROC-кривая и AUC
        rf_proba = rf.predict_proba(X_test_st)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, rf_proba)
        auc_score = roc_auc_score(y_test, rf_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        ax2.plot([0, 1], [0, 1], linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------------------------------------------------------------------
        # -------------------------------ГРАДИЕНТНЫЙ БУСТИНГ-----------------------------------
        # -------------------------------------------------------------------------------------

        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(X_train_st, y_train)
        pred_xgb = xgb.predict(X_test_st)

        # Оценка модели Градиентного бустинга
        accuracy_xgb = accuracy_score(y_test, pred_xgb)
        conf_matrix_xgb = confusion_matrix(y_test, pred_xgb)
        classification_rep_xgb = classification_report(y_test, pred_xgb)

        # Визуализация результатов
        st.header("Результаты обучения модели Градиентного бустинга")
        st.write(f"Accuracy: {accuracy_xgb:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_rep_xgb)

        # ROC-кривая и AUC
        xgb_proba = xgb.predict_proba(X_test_st)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, xgb_proba)
        auc_score = roc_auc_score(y_test, xgb_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        ax2.plot([0, 1], [0, 1], linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------------------------------------------------------------------
        # -------------------------------МЕТОД ОПОРНЫХ ВЕКТОРОВ--------------------------------
        # -------------------------------------------------------------------------------------

        svc = SVC(probability=True)
        svc.fit(X_train_st, y_train)
        pred_svc = svc.predict(X_test_st)

        # Оценка модели Метода опорных векторов
        accuracy_svc = accuracy_score(y_test, pred_svc)
        conf_matrix_svc = confusion_matrix(y_test, pred_svc)
        classification_rep_svc = classification_report(y_test, pred_svc)

        # Визуализация результатов
        st.header("Результаты обучения модели Метода опорных векторов")
        st.write(f"Accuracy: {accuracy_svc:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_svc, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_rep_svc)

        # ROC-кривая и AUC
        svc_proba = svc.predict_proba(X_test_st)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, svc_proba)
        auc_score = roc_auc_score(y_test, svc_proba)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
        ax2.plot([0, 1], [0, 1], linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------------------------------------------------------------------
        # -------------------------------STREAMLIT-ПРИЛОЖЕНИЕ----------------------------------
        # -------------------------------------------------------------------------------------

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

            # -------------------------------------------------------------------------------------
            # ----------------------------------ПРЕДСКАЗАНИЕ---------------------------------------
            # -------------------------------------------------------------------------------------

            # Создаем список, для того, чтобы произвести проверку на выявление лучшего результата
            accuracy = [accuracy_y, accuracy_rf, accuracy_xgb, accuracy_svc]

            # Проверку по выявлению наилучшего результата
            if max(accuracy) == accuracy_y:
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)[:, 1]

                st.write(f"Предсказание LR(0 - без отказа, 1 - отказ): {prediction[0]}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")

            if max(accuracy) == accuracy_rf:
                prediction = rf.predict(input_df)
                prediction_proba = rf.predict_proba(input_df)[:, 1]

                st.write(f"Предсказание RF(0 - без отказа, 1 - отказ): {prediction[0]}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")

            if max(accuracy) == accuracy_xgb:
                prediction = xgb.predict(input_df)
                prediction_proba = xgb.predict_proba(input_df)[:, 1]

                st.write(f"Предсказание XGB(0 - без отказа, 1 - отказ): {prediction[0]}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")

            if max(accuracy) == accuracy_svc:
                prediction = svc.predict(input_df)
                prediction_proba = svc.predict_proba(input_df)[:, 1]

                st.write(f"Предсказание SVC(0 - без отказа, 1 - отказ): {prediction[0]}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")