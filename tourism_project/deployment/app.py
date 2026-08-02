from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/wellness_package_model.joblib")

st.set_page_config(
    page_title="Wellness Package Predictor",
    page_icon="✈️",
    layout="wide"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run the GitHub Actions training pipeline first."
        )
    return joblib.load(MODEL_PATH)


st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Estimate whether a customer is likely to purchase the "
    "Wellness Tourism Package."
)

model = load_model()

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 35)
        type_of_contact = st.selectbox(
            "Type of Contact",
            ["Company Invited", "Self Enquiry"]
        )
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration_of_pitch = st.number_input(
            "Duration of Pitch", 0.0, 120.0, 15.0
        )
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting", 1, 20, 2
        )
        number_of_followups = st.number_input(
            "Number of Follow-ups", 0.0, 20.0, 3.0
        )
        product_pitched = st.selectbox(
            "Product Pitched",
            ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
        )
        preferred_property_star = st.selectbox(
            "Preferred Property Star", [3.0, 4.0, 5.0]
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )
        number_of_trips = st.number_input(
            "Annual Number of Trips", 0.0, 50.0, 2.0
        )

    with col3:
        passport = st.selectbox(
            "Has Passport", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        pitch_satisfaction_score = st.selectbox(
            "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
        )
        own_car = st.selectbox(
            "Owns Car", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        number_of_children_visiting = st.number_input(
            "Children Visiting", 0.0, 10.0, 0.0
        )
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthly_income = st.number_input(
            "Monthly Income", 0.0, 1000000.0, 25000.0
        )

    submitted = st.form_submit_button("Predict Purchase Likelihood")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    st.subheader("Input record")
    st.dataframe(input_df, use_container_width=True)

    st.metric("Predicted purchase probability", f"{probability:.1%}")

    if prediction == 1:
        st.success("Likely buyer — prioritize this customer for contact.")
    else:
        st.info("Lower purchase likelihood — use a lower-priority campaign.")
