import streamlit as st
import pandas as pd
import joblib
import base64
import sklearn


st.set_page_config(page_title=" Employee Attrition Predictor", layout="centered",page_icon="🧑‍💻")


# Add background image CSS
background_url = "https://images.shiksha.com/mediadata/ugcDocuments/images/wordpressImages/2023_08_Employee-retention.jpg"
page_bg_img = f"""
<style>
.stApp {{
    position: relative;
    background-image: url("{background_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Add a semi-transparent overlay */
.stApp::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);  /* black with 40% opacity */
    z-index: 0;
}}

.stApp > * {{
    position: relative;
    z-index: 1;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)



model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
top15_features = joblib.load("top15_features.pkl")
model_features = joblib.load("model_features.pkl")
ohe = joblib.load("onehot_encoder.pkl")

# set title
st.markdown(
    """
    <h2 style='text-align: center; color: orange; font-size: 32px; font-weight : bold;'>
        🔍 EMPLOYEE ATTRITION PREDICTOR
    </h2>""",
    unsafe_allow_html=True
)
st.markdown( "<i>Fill in the details below to predict whether an employee is likely to leave the company.</i>",unsafe_allow_html=True)

with st.form("attrition_form"):
  st.markdown("### 🧑‍💻 <span style='color:#ff4b4b'>Employee Details</span>", unsafe_allow_html=True)

  StockOptionLevel = st.selectbox("📦 **Stock Option Level**", [0, 1, 2, 3])
  MonthlyIncome = st.number_input("💰 **Monthly Income**", min_value=1000, max_value=20000, value=5000)
  JobSatisfaction = st.slider("😊 **Job Satisfaction**", 1, 4, 3)
  YearsWithCurrManager = st.slider("🧑‍💼 **Years with Current Manager**", 0, 20, 5)
  RelationshipSatisfaction = st.slider("👥 **Relationship Satisfaction**", 1, 4, 3)
  JobInvolvement = st.slider("🔨 **Job Involvement**", 1, 4, 3)
  YearsAtCompany = st.slider("🏢 **Years at Company**", 0, 40, 5)
  TotalWorkingYears = st.slider("📅 **Total Working Years**", 0, 40, 10)
  MonthlyRate = st.slider("📈 **Monthly Rate**", 1000, 30000, 500)
  Age = st.slider("🎂 **Age**", 18, 60, 30)
  DailyRate = st.number_input("📊 **Daily Rate**", min_value=100, max_value=1500, value=800)

  BusinessTravel = st.selectbox("✈️ **Business Travel**", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
  Department = st.selectbox("🏢 **Department**", ["Research & Development", "Sales", "Human Resources"])
  EducationField = st.selectbox("🎓 **Education Field**", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
  JobRole = st.selectbox("🧑‍💻 **Job Role**", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                                                "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
  MaritalStatus = st.selectbox("💍 **Marital Status**", ["Single", "Married", "Divorced"])

  submit = st.form_submit_button("🔍 Predict Attrition")

if submit:
  input_dict = {"StockOptionLevel": StockOptionLevel,
    "MonthlyIncome": MonthlyIncome,
    "JobSatisfaction": JobSatisfaction,
    "YearsWithCurrManager": YearsWithCurrManager,
    "RelationshipSatisfaction": RelationshipSatisfaction,
    "JobInvolvement": JobInvolvement,
    "YearsAtCompany": YearsAtCompany,
    "TotalWorkingYears": TotalWorkingYears,
    "MonthlyRate": MonthlyRate,
    "Age": Age,
    "DailyRate": DailyRate,
    "BusinessTravel": BusinessTravel,
    "Department": Department,
    "EducationField": EducationField,
    "JobRole": JobRole,
    "MaritalStatus": MaritalStatus}

  input_df = pd.DataFrame([input_dict])

  multi_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
  encoded_cats = ohe.transform(input_df[multi_cols])
  input_df = input_df.drop(columns=multi_cols).reset_index(drop=True)
  input_encoded = pd.concat([input_df, encoded_cats], axis=1)

  for col in model_features:
      if col not in input_encoded.columns:
          input_encoded[col] = 0
  input_encoded = input_encoded[model_features]


  input_scaled = pd.DataFrame(scaler.transform(input_encoded), columns=model_features)
  input_final = input_scaled[top15_features]

  prediction = model.predict(input_final)[0]

  st.markdown("---")
  if prediction == 1:
      st.markdown(
          "<div class='result-box' style='background-color:#ffe6e6; color:#cc0000;'>⚠️ <b>Prediction:</b> The employee is <u>likely to leave</u> the company.</div>",
          unsafe_allow_html=True)
  else:
      st.markdown(
          "<div class='result-box' style='background-color:#e6ffed; color:#006600;'>✅ <b>Prediction:</b> The employee is <u>likely to stay</u> with the company.</div>",
          unsafe_allow_html=True)
