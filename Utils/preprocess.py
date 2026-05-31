import pandas as pd

def preprocess(user_df):

    ordinal_mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2
    }

    peer_mapping = {
        'Negative': 0,
        'Neutral': 1,
        'Positive': 2
    }

    yes_no_mapping = {
        'No': 0,
        'Yes': 1
    }

    gender_mapping = {
        'Female': 0,
        'Male': 1
    }

    df = user_df.copy()

    ordinal_columns = [
        'Parental_Involvement',
        'Access_to_Resources',
        'Motivation_Level',
        'Teacher_Quality',
        'Family_Income'
    ]

    for column in ordinal_columns:
        df[column] = df[column].map(ordinal_mapping)

    df['Peer_Influence'] = df['Peer_Influence'].map(peer_mapping)

    binary_columns = [
        'Extracurricular_Activities',
        'Internet_Access',
        'Learning_Disabilities'
    ]

    for column in binary_columns:
        df[column] = df[column].map(yes_no_mapping)

    df['Gender'] = df['Gender'].map(gender_mapping)

    df['School_Type_Public'] = (
        df['School_Type'] == 'Public'
    ).astype(int)

    df['Parental_Education_Level_High School'] = (
        df['Parental_Education_Level'] == 'High School'
    ).astype(int)

    df['Parental_Education_Level_Postgraduate'] = (
        df['Parental_Education_Level'] == 'Postgraduate'
    ).astype(int)

    df['Distance_from_Home_Moderate'] = (
        df['Distance_from_Home'] == 'Moderate'
    ).astype(int)

    df['Distance_from_Home_Near'] = (
        df['Distance_from_Home'] == 'Near'
    ).astype(int)

    df = df.drop(
        [
            'School_Type',
            'Parental_Education_Level',
            'Distance_from_Home'
        ],
        axis=1
    )

    feature_order = [
        'Hours_Studied',
        'Attendance',
        'Parental_Involvement',
        'Access_to_Resources',
        'Extracurricular_Activities',
        'Sleep_Hours',
        'Previous_Scores',
        'Motivation_Level',
        'Internet_Access',
        'Tutoring_Sessions',
        'Family_Income',
        'Teacher_Quality',
        'Peer_Influence',
        'Physical_Activity',
        'Learning_Disabilities',
        'Gender',
        'Exam_Score',
        'School_Type_Public',
        'Parental_Education_Level_High School',
        'Parental_Education_Level_Postgraduate',
        'Distance_from_Home_Moderate',
        'Distance_from_Home_Near'
    ]

    return df[feature_order]