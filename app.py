import streamlit as st
import pandas as pd
import joblib

from Utils.preprocess import preprocess

classifier = joblib.load("Models/classifier.pkl")
regressor = joblib.load("Models/regressor.pkl")

st.set_page_config(
    page_title="Grant AI System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Grant AI System")
st.write(
    "Talaba ma'lumotlari asosida grant olish ehtimoli va grant miqdorini aniqlash tizimi."
)


st.header("📚 Academic Information")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=50,
        value=20
    )

    attendance = st.number_input(
        "Attendance",
        min_value=0,
        max_value=100,
        value=80
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0,
        max_value=24,
        value=8
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0,
        max_value=100,
        value=70
    )

with col2:
    exam_score = st.number_input(
        "Exam Score",
        min_value=0,
        max_value=100,
        value=70
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions",
        min_value=0,
        max_value=20,
        value=2
    )

    physical_activity = st.number_input(
        "Physical Activity",
        min_value=0,
        max_value=20,
        value=3
    )


st.header("🏠 Family Information")

col1, col2 = st.columns(2)

with col1:
    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

    parental_education_level = st.selectbox(
        "Parental Education Level",
        ["College", "High School", "Postgraduate"]
    )

with col2:
    access_to_resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )

    distance_from_home = st.selectbox(
        "Distance from Home",
        ["Far", "Moderate", "Near"]
    )

st.header("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    motivation_level = st.selectbox(
    "Motivation Level",
    ["Low", "Medium", "High"]
    )


    peer_influence = st.selectbox(
        "Peer Influence",
        ["Negative", "Neutral", "Positive"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


with col2:
    extracurricular_activities = st.selectbox(
    "Extracurricular Activities",
    ["No", "Yes"]
    )


    internet_access = st.selectbox(
        "Internet Access",
        ["No", "Yes"]
    )

    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["No", "Yes"]
    )

    school_type = st.selectbox(
    "School Type",
    ["Private", "Public"]
    )


if st.button("🔍 Predict Grant"):

    user_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Extracurricular_Activities': extracurricular_activities,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Tutoring_Sessions': tutoring_sessions,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'Peer_Influence': peer_influence,
        'Physical_Activity': physical_activity,
        'Learning_Disabilities': learning_disabilities,
        'Gender': gender,
        'Exam_Score': exam_score,
        'School_Type': school_type,
        'Parental_Education_Level': parental_education_level,
        'Distance_from_Home': distance_from_home
    }

    user_df = pd.DataFrame([user_data])

    processed_df = preprocess(user_df)

    grant_prediction = classifier.predict(processed_df)[0]

    if grant_prediction == 0:

        st.error("❌ Talaba grant olish huquqiga ega emas.")

    else:

        grant_percent = regressor.predict(processed_df)[0]

        grant_percent = round(grant_percent)

        grant_amount = int(
            (grant_percent / 100) * 39000000
        )

        st.success("✅ Talaba grant olish huquqiga ega.")

        st.subheader("Natija")

        st.write(f"Grant Foizi: {grant_percent}%")

        st.write(
            f"Grant Miqdori: {grant_amount:,} so'm"
        )