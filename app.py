import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("Israel-Palestine Conflict Analysis Dashboard")

st.sidebar.title("Upload Incident Dataset (CSV)")
upload_file = st.sidebar.file_uploader("Choose a CSV file", type='csv')

if upload_file is not None:
    df = pd.read_csv(upload_file)

    # Sidebar Summary
    st.sidebar.write("### Quick Stats")
    st.sidebar.write("**Number of Events:**", len(df))
    st.sidebar.write("**Citizenship Counts:**", df['citizenship'].value_counts())
    st.sidebar.write("**Event Locations:**", df['event_location_region'].value_counts())
    st.sidebar.write("**Weapons Used:**", df['ammunition'].value_counts())

    st.write("## 📊 Raw Data Preview")
    st.dataframe(df.head())

    st.write("## 🧍 Gender Distribution")
    st.bar_chart(df['gender'].value_counts())

    st.write("## 🎯 Injury Types")
    st.bar_chart(df['type_of_injury'].value_counts())

    st.write("## 📍 Event Location Counts")
    st.bar_chart(df['event_location_region'].value_counts())

    st.write("## 🕒 Events Over Time")
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='coerce')
    df['year'] = df['date_of_event'].dt.year
    df['month'] = df['date_of_event'].dt.month_name()
    time_events = df.groupby(['year', 'month']).size().reset_index(name='count')
    time_events['year_month'] = time_events['month'] + " " + time_events['year'].astype(str)
    st.line_chart(time_events.set_index('year_month')['count'])

    st.write("## 🧠 Simple ML Analysis: Predict Hostility Participation")
    if 'took_part_in_the_hostilities' in df.columns and 'age' in df.columns and 'gender' in df.columns:
        # Encode gender
        df_ml = df[['age', 'gender', 'took_part_in_the_hostilities']].dropna()
        df_ml['gender'] = df_ml['gender'].map({'Male': 0, 'Female': 1})
        df_ml['target'] = df_ml['took_part_in_the_hostilities'].map({'Yes': 1, 'No': 0})

        X = df_ml[['age', 'gender']]
        y = df_ml['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write("### Model Accuracy Report")
        st.text(classification_report(y_test, preds))
    else:
        st.warning("Dataset must contain 'age', 'gender' and 'took_part_in_the_hostilities' columns for ML prediction.")
