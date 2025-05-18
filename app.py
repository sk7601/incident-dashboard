import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.sidebar.title("Upload dataset")
upload_file = st.sidebar.file_uploader("Choose CSV file", type='csv')
if upload_file is not None:
    df = pd.read_csv(upload_file)

    # sidebar summary
    st.sidebar.write("Number of Events:", len(df))
    st.sidebar.subheader("Citizenship Count")
    st.sidebar.write(df['citizenship'].value_counts())
    st.sidebar.subheader("Weapons Used")
    st.sidebar.write(df['ammunition'].value_counts())

    st.title("Israel-Palestine Conflict Analysis Dashboard")
    st.write("Dataset Sample", df.head())

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Type of Injuries")
        st.bar_chart(df['type_of_injury'].value_counts())
    with col2:
        st.subheader("Gender Count")
        st.bar_chart(df['gender'].value_counts())

    # Age stats
    st.subheader("Age Summary")
    st.write(df['age'].describe())

    # Pie charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Residence by Region")
        region_residence = df.groupby('event_location_region')['place_of_residence'].nunique()
        fig, ax = plt.subplots()
        ax.pie(region_residence, labels=region_residence.index, autopct='%1.1f%%')
        st.pyplot(fig)
    with col2:
        st.subheader("Injury Types Pie")
        injury = df['type_of_injury'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(injury, labels=injury.index, autopct='%1.1f%%')
        st.pyplot(fig)

    # Time-based event chart
    df['date_of_event'] = pd.to_datetime(df['date_of_event'], errors='coerce')
    df.dropna(subset=['date_of_event'], inplace=True)
    df['year'] = df['date_of_event'].dt.year
    df['month'] = df['date_of_event'].dt.month_name()
    timeline = df.groupby(['year', 'month']).size().reset_index(name='count')
    timeline['year_month'] = timeline['month'] + ' ' + timeline['year'].astype(str)
    st.subheader("Time-Based Events")
    st.line_chart(timeline.set_index('year_month')['count'])

    # -------------------------
    # 🧠 Machine Learning Part
    # -------------------------
    st.subheader("ML Prediction: Hostilities Participation")

    df_ml = df[['age', 'gender', 'citizenship', 'type_of_injury', 'took_part_in_the_hostilities']].dropna()

    # Encode categorical values
    encoders = {}
    for col in ['gender', 'citizenship', 'type_of_injury', 'took_part_in_the_hostilities']:
        enc = LabelEncoder()
        df_ml[col] = enc.fit_transform(df_ml[col])
        encoders[col] = enc

    X = df_ml.drop('took_part_in_the_hostilities', axis=1)
    y = df_ml['took_part_in_the_hostilities']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Take input from user
    st.markdown("### Predict Hostility Involvement")
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    gender = st.selectbox("Gender", encoders['gender'].classes_)
    citizenship = st.selectbox("Citizenship", encoders['citizenship'].classes_)
    injury = st.selectbox("Type of Injury", encoders['type_of_injury'].classes_)

    input_data = pd.DataFrame([[
        age,
        encoders['gender'].transform([gender])[0],
        encoders['citizenship'].transform([citizenship])[0],
        encoders['type_of_injury'].transform([injury])[0]
    ]], columns=['age', 'gender', 'citizenship', 'type_of_injury'])

    prediction = clf.predict(input_data)[0]
    prediction_label = encoders['took_part_in_the_hostilities'].inverse_transform([prediction])[0]

    prediction_proba = clf.predict_proba(input_data)[0]
    class_labels = clf.classes_
    proba_for_prediction = prediction_proba[list(class_labels).index(prediction)]

    st.write(f"Prediction: **{prediction_label}**")
    st.write(f"Confidence: **{proba_for_prediction:.2f}**")

