import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Sidebar - Upload CSV
st.sidebar.title("Upload dataset")
upload_file = st.sidebar.file_uploader("Choose CSV file", type='csv')
if upload_file is not None:
    df = pd.read_csv(upload_file)

    # Sidebar Data Summary
    st.sidebar.write("No of Events:", len(df))
    st.sidebar.subheader("Citizenship Counts")
    st.sidebar.write(df['citizenship'].value_counts())
    st.sidebar.subheader("Event Location Region")
    st.sidebar.write(df['event_location_region'].value_counts())
    st.sidebar.subheader("Hostilities (Yes)")
    st.sidebar.write(df[df['took_part_in_the_hostilities'] == 'Yes']['citizenship'].value_counts())
    st.sidebar.subheader("Hostilities (No)")
    st.sidebar.write(df[df['took_part_in_the_hostilities'] == 'No']['citizenship'].value_counts())
    st.sidebar.subheader("Weapons Used")
    st.sidebar.write(df['ammunition'].value_counts())

    # Main Title
    st.title("Israel-Palestine Conflict Dashboard")
    st.write("Sample Dataset")
    st.write(df.head())

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Type of Injuries")
        st.bar_chart(df['type_of_injury'].value_counts())
    with col2:
        st.subheader("Gender Count")
        st.bar_chart(df['gender'].value_counts())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Summary")
        st.write(df['age'].describe())
    with col2:
        st.subheader("Event Regions")
        st.bar_chart(df['event_location_region'].value_counts())

    # Pie Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Residence % by Region")
        residence = df.groupby('event_location_region')['place_of_residence'].nunique()
        fig1, ax1 = plt.subplots()
        ax1.pie(residence, labels=residence.index, autopct='%1.1f%%')
        st.pyplot(fig1)
    with col2:
        st.subheader("Injury Type Distribution")
        injurytype = df['type_of_injury'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(injurytype, labels=injurytype.index, autopct='%1.1f%%')
        st.pyplot(fig2)

    st.subheader("Average Age by Region")
    avg_age = df.groupby('event_location_region')['age'].mean()
    st.bar_chart(avg_age)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Incidents by Citizenship")
        st.write(df.groupby('citizenship').size().reset_index(name='incident_count'))
    with col2:
        st.subheader("Incidents by Gender")
        st.write(df.groupby('gender').size().reset_index(name='incident_count'))

    # Time Analysis
    df['date_of_event'] = pd.to_datetime(df['date_of_event'])
    df['year'] = df['date_of_event'].dt.year
    df['month'] = df['date_of_event'].dt.month_name()
    time_events = df.groupby(['year', 'month']).size().reset_index(name='incident_count')
    time_events['year_month'] = time_events['month'] + ' ' + time_events['year'].astype(str)
    st.subheader("Time-based Incident Trend")
    st.line_chart(time_events.set_index('year_month')['incident_count'])

    # ----------------- ML 1: Predict Hostility Involvement -----------------
    st.header("ML: Predict Hostility Participation")
    ml_df = df[['age', 'gender', 'citizenship', 'ammunition', 'took_part_in_the_hostilities']].dropna()
    for col in ['gender', 'citizenship', 'ammunition', 'took_part_in_the_hostilities']:
        ml_df[col] = LabelEncoder().fit_transform(ml_df[col])
    X = ml_df.drop('took_part_in_the_hostilities', axis=1)
    y = ml_df['took_part_in_the_hostilities']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    st.success(f"Hostility Model Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")

    st.subheader("Predict Hostility")
    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", df['gender'].dropna().unique())
    citizenship = st.selectbox("Citizenship", df['citizenship'].dropna().unique())
    ammo = st.selectbox("Ammunition", df['ammunition'].dropna().unique())

    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'citizenship': [citizenship],
        'ammunition': [ammo]
    })
    for col in ['gender', 'citizenship', 'ammunition']:
        le = LabelEncoder()
        le.fit(df[col].dropna())
        input_data[col] = le.transform(input_data[col])
    prediction = clf.predict(input_data)[0]
    pred_label = "Yes" if prediction == 1 else "No"
    st.info(f"Predicted: Took Part in Hostilities? → **{pred_label}**")

    # ----------------- ML 2: Predict Injury Type -----------------
    st.header("ML: Predict Injury Type")
    inj_df = df[['age', 'gender', 'citizenship', 'ammunition', 'type_of_injury']].dropna()
    for col in ['gender', 'citizenship', 'ammunition', 'type_of_injury']:
        inj_df[col] = LabelEncoder().fit_transform(inj_df[col])
    X2 = inj_df.drop('type_of_injury', axis=1)
    y2 = inj_df['type_of_injury']
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2)
    clf_injury = RandomForestClassifier()
    clf_injury.fit(X2_train, y2_train)

    st.success(f"Injury Type Model Accuracy: {clf_injury.score(X2_test, y2_test) * 100:.2f}%")

    st.subheader("Predict Injury Type")
    age2 = st.slider("Age", 1, 100, 30, key='inj')
    gender2 = st.selectbox("Gender", df['gender'].dropna().unique(), key='inj')
    citizenship2 = st.selectbox("Citizenship", df['citizenship'].dropna().unique(), key='inj')
    ammo2 = st.selectbox("Ammunition", df['ammunition'].dropna().unique(), key='inj')
    input_df2 = pd.DataFrame({
        'age': [age2],
        'gender': [gender2],
        'citizenship': [citizenship2],
        'ammunition': [ammo2]
    })
    for col in ['gender', 'citizenship', 'ammunition']:
        le = LabelEncoder()
        le.fit(df[col].dropna())
        input_df2[col] = le.transform(input_df2[col])
    pred_injury = clf_injury.predict(input_df2)[0]
    injury_label = LabelEncoder().fit(df['type_of_injury'].dropna()).inverse_transform([pred_injury])[0]
    st.info(f"Predicted Injury Type: **{injury_label}**")

    # ----------------- Clustering -----------------
    st.header("Incident Clustering")
    cluster_df = df[['age', 'gender', 'citizenship', 'ammunition']].dropna()
    for col in ['gender', 'citizenship', 'ammunition']:
        cluster_df[col] = LabelEncoder().fit_transform(cluster_df[col])
    kmeans = KMeans(n_clusters=3)
    cluster_df['Cluster'] = kmeans.fit_predict(cluster_df)
    st.write(cluster_df.head())
    st.bar_chart(cluster_df['Cluster'].value_counts())

    # ----------------- Feature Importance -----------------
    st.header("Feature Importance (Hostility Prediction)")
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write(importance_df)
    fig3, ax3 = plt.subplots()
    ax3.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
    st.pyplot(fig3)
