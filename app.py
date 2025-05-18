import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

st.sidebar.title("Upload dataset")
upload_file = st.sidebar.file_uploader("Choose CSV file", type='csv')

if upload_file is not None:
    df = pd.read_csv(upload_file)

    # Show basic info and sidebar stats
    no_event = len(df)
    citizenship_counts = df['citizenship'].value_counts()
    event_location_region = df['event_location_region'].value_counts()
    hostilities_counts = df[df['took_part_in_the_hostilities'] == 'Yes']['citizenship'].value_counts()
    no_hostilities_counts = df[df['took_part_in_the_hostilities'] == 'No']['citizenship'].value_counts()

    st.sidebar.write("No of Events:", no_event)

    col1, col2 = st.sidebar.columns(2)
    col3, col4 = st.sidebar.columns(2)
    with col1:
        st.subheader("Citizenship Counts")
        st.write(citizenship_counts)
    with col2:
        st.subheader("Event Location Region")
        st.write(event_location_region)
    with col3:
        st.subheader("Hostilities Counts")
        st.write(hostilities_counts)
    with col4:
        st.subheader("No Hostilities Counts")
        st.write(no_hostilities_counts)

    weapons_counts = df['ammunition'].value_counts()
    st.sidebar.write("Weapon counts", weapons_counts)

    # Data Analysis part
    st.title("Israel Palestine Conflict Analysis")
    st.write('Dataset Sample', df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Type of Injuries")
        type_of_injury = df['type_of_injury'].value_counts()
        st.bar_chart(type_of_injury)
    with col2:
        st.subheader("Male/Female Count")
        MFcounts = df['gender'].value_counts()
        st.bar_chart(MFcounts, color='#FF0000')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Summary")
        age = df['age'].describe()
        st.write(age)
    with col2:
        st.subheader("Event Location Region Count")
        eventregion = df['event_location_region'].value_counts()
        st.bar_chart(eventregion)

    col1, col2 = st.columns(2)
    with col1:
        residencecountbyreg = df.groupby('event_location_region')['place_of_residence'].nunique()
        st.subheader("Residence Percentage by Region")
        fig, ax = plt.subplots()
        ax.pie(residencecountbyreg, labels=residencecountbyreg.index, autopct='%1.1f%%')
        st.pyplot(fig)
    with col2:
        injurytype = df['type_of_injury'].value_counts()
        st.subheader("Injury Types")
        fig, ax = plt.subplots()
        ax.pie(injurytype, labels=injurytype.index, autopct='%1.1f%%')
        st.pyplot(fig)

    regionavgage = df.groupby('event_location_region')['age'].mean()
    st.subheader("Avg Age by Region")
    st.bar_chart(regionavgage)

    col1, col2 = st.columns(2)
    with col1:
        IncidentcountbyNat = df.groupby('citizenship').size().reset_index(name='incident_count')
        st.subheader('Incident Count by Nationality')
        st.write(IncidentcountbyNat)
    with col2:
        genderInc = df.groupby('gender').size().reset_index(name="incident_count")
        st.subheader('Incident Count by Gender')
        st.write(genderInc)

    # Time-based analysis (events at specific times)
    df['date_of_event'] = pd.to_datetime(df['date_of_event'])
    df['year'] = df['date_of_event'].dt.year
    df['month'] = df['date_of_event'].dt.month_name()
    time_events = df.groupby(['year', 'month']).size().reset_index(name='incident_count')
    time_events['year_month'] = time_events['month'] + ' ' + time_events['year'].astype(str)
    st.subheader('Time-Based Events')
    st.line_chart(time_events.set_index('year_month')['incident_count'])

    # === ML Model Part ===
    st.subheader("ML Model: Predict Participation in Hostilities")

    # Prepare data for ML
    df_ml = df.copy()

    # Encode target: Yes = 1, No = 0
    df_ml = df_ml[df_ml['took_part_in_the_hostilities'].isin(['Yes', 'No'])]
    df_ml['target'] = df_ml['took_part_in_the_hostilities'].map({'Yes': 1, 'No': 0})

    # Features we use: age, gender, citizenship, type_of_injury
    features = ['age', 'gender', 'citizenship', 'type_of_injury']
    df_ml = df_ml.dropna(subset=features + ['target'])

    # Encode categorical features
    le_gender = LabelEncoder()
    le_citizenship = LabelEncoder()
    le_injury = LabelEncoder()

    df_ml['gender_enc'] = le_gender.fit_transform(df_ml['gender'])
    df_ml['citizenship_enc'] = le_citizenship.fit_transform(df_ml['citizenship'])
    df_ml['injury_enc'] = le_injury.fit_transform(df_ml['type_of_injury'])

    X = df_ml[['age', 'gender_enc', 'citizenship_enc', 'injury_enc']]
    y = df_ml['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict & accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: **{acc * 100:.2f}%**")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Interactive prediction input
    st.subheader("Try it yourself")

    input_age = st.number_input("Age", min_value=0, max_value=120, value=25)
    input_gender = st.selectbox("Gender", le_gender.classes_)
    input_citizenship = st.selectbox("Citizenship", le_citizenship.classes_)
    input_injury = st.selectbox("Type of Injury", le_injury.classes_)

    # Encode input
    input_data = pd.DataFrame({
        'age': [input_age],
        'gender_enc': le_gender.transform([input_gender]),
        'citizenship_enc': le_citizenship.transform([input_citizenship]),
        'injury_enc': le_injury.transform([input_injury])
    })

    prediction = clf.predict(input_data)[0]
    prediction_proba = clf.predict_proba(input_data)[0][prediction]

    result = "Yes" if prediction == 1 else "No"
    st.write(f"Prediction: Took Part in Hostilities? **{result}** (Confidence: {prediction_proba * 100:.2f}%)")
