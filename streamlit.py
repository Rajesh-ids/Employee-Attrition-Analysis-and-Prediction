import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns  
import plotly.graph_objects as go  
from scipy.stats import chi2_contingency  
from sklearn.preprocessing import MinMaxScaler  
from streamlit_option_menu import option_menu  

# Load models and scaler
with open("D:\project\EmployeeAttrition\EmployeePromotionLikelihood1.pkl", 'rb') as file:
    promotion_model = pickle.load(file)

with open("D:\project\EmployeeAttrition\performance_rating_model.pkl",'rb') as file:
    performance_model = pickle.load(file)

with open("D:\project\EmployeeAttrition\Attrition_rate1.pkl", 'rb') as file:
    attrition_model = pickle.load(file)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data for EDA
data = pd.read_excel("D:\project\EmployeeAttrition\Employee-Attrition.xlsx")

# Set page configuration
st.set_page_config(page_title="Employee Portal", layout="wide")

# Side Navigation Bar
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=[
            "Attrition Prediction",
            "Performance Rating Prediction",
            "Employee Promotion Likelihood"
        ],
        icons=["person-x", "bar-chart-line", "arrow-up-circle"]
    )

# Page 1: Attrition Prediction
if selected == "Attrition Prediction":
    st.title("❗ Attrition Risk Prediction")
    JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    OverTime = st.selectbox("OverTime", ['No', 'Yes'])
    Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    JobLevel = st.number_input("Job Level", 1, 5)
    YearsAtCompany = st.number_input("Years at Company", 0, 40)
    YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40)

    if st.button("Predict Attrition Risk"):
        OverTime_encoded = 1 if OverTime == 'Yes' else 0
        input_data = [[JobSatisfaction, OverTime_encoded, Education, JobLevel, YearsAtCompany, YearsInCurrentRole, TotalWorkingYears]]
        prediction = attrition_model.predict(input_data)[0]
        result = "YES" if prediction == 1 else "NO"
        st.success(f"Attrition Risk: {result}")

# Page 2: Performance Rating Prediction
elif selected == "Performance Rating Prediction":
    st.title("📈 Performance Rating Prediction")
    YearsAtCompany = st.number_input("Years at Company", 0, 40)
    Education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20)
    YearsWithCurrManager = st.number_input("Years with Current Manager", 0, 20)
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 20)
    JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])

    if st.button("Predict Performance Rating"):
        input_data = [[YearsAtCompany, Education, YearsInCurrentRole, YearsWithCurrManager, YearsSinceLastPromotion, JobInvolvement]]
        prediction = performance_model.predict(input_data)[0]
        st.success(f"Predicted Performance Rating: {prediction}")

# Page 3: Employee Promotion Likelihood
elif selected == "Employee Promotion Likelihood":
    st.title("🚀 Employee Promotion Likelihood")
    JobLevel = st.number_input("Job Level", 1, 5)
    TotalWorkingYears = st.number_input("Total Working Years", 0, 40)
    YearsInCurrentRole = st.number_input("Years in Current Role", 0, 20)
    MonthlyIncome = st.number_input("MonthlyIncome", 0, 20000)
    YearsAtCompany = st.selectbox("Years At Company", [1, 2, 3, 4, 5, 6, 7, 8])
    YearsWithCurrManager = st.number_input("Years with Current Manager", 0, 10)

    if st.button("Predict Promotion Time"):
        input_data = [[JobLevel, TotalWorkingYears, YearsInCurrentRole, MonthlyIncome, YearsAtCompany, YearsWithCurrManager]]
        scaled_input = scaler.transform(input_data)
        prediction = promotion_model.predict(scaled_input)[0]
        st.success(f"Estimated Years Since Last Promotion: {prediction} years")




#  <-- css --> #

# Set background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSCPlPGLcHvQVSEo8UIR-OF_NN2h0PAJ_Fgw&s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# set input box color
st.markdown(
    """
    <style>
    /* Blue border for selectbox */
    div[data-baseweb="select"] > div {
        border: 2px solid blue !important;
        border-radius: 5px;
    }

    div[data-baseweb="select"]:focus-within > div {
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.5) !important;
    }

    /* Blue border for number_input */
    input[type="number"] {
        border: 2px solid blue !important;
        border-radius: 5px;
        padding: 5px;
    }

    input[type="number"]:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


