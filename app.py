import streamlit as st
import joblib
import pandas as pd



model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


st.title("🎬 Movie Like Prediction App")

st.write("Predict whether a user will like a movie")


age = st.slider("Age", 13, 65)

watchtime = st.slider("Watch Time Per Day (hours)", 0.5, 6.0)

genre = st.selectbox(
    "Preferred Genre",
    ["Action", "Drama", "Comedy", "Horror", "Sci-Fi", "Romance", "Thriller"]
)

duration = st.slider("Movie Duration (minutes)", 80, 200)

year = st.slider("Release Year", 1990, 2025)

rating = st.slider("Movie Rating", 4.0, 9.5)

popular = st.selectbox("Is Popular", [0, 1])

device = st.selectbox(
    "Device",
    ["Mobile", "TV", "Laptop"]
)

subscription = st.selectbox(
    "Subscription Type",
    ["Basic", "Premium"]
)



if st.button("Predict"):

    input_data = pd.DataFrame([{

        "Age": age,
        "WatchTimePerDay": watchtime,
        "PreferredGenre": genre,
        "MovieDuration": duration,
        "ReleaseYear": year,
        "Rating": rating,
        "IsPopular": popular,
        "Device": device,
        "SubscriptionType": subscription

    }])


    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)[0]


    if prediction == 1:

        st.success("✅ User will LIKE the movie")

    else:

        st.error("❌ User will NOT LIKE the movie")