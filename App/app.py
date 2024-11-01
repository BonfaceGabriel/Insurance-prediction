import streamlit as st
import pandas as pd
import joblib
import lightgbm



data = pd.read_csv('../Nakuru_FinAccess.csv')

#load models
classifier = joblib.load('./models/classifier.joblib')
cluster = joblib.load('./models/kproto.joblib')
 
#title
st.title('Insurance Product Prediction App')

#sidebar
st.sidebar.title('User Input Features')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        income_source = st.sidebar.selectbox('Source of Income',('Casual_work','Family','Agriculture', 'Business', 'Employment', 'Renting', 'Pension', 'Aid'))
        most_important_life_goal = st.sidebar.selectbox('Most Important Life Goal',('Food','Education', 'Health', 'Business', 'Career', 'Home', 'Assets'))
        nearest_financial_prod = st.sidebar.selectbox('Nearest Financial Product',('MMoney','Bank', 'Insurance'))
        area = st.sidebar.selectbox('Area',('Urban','Rural'))
        avg_mnth_income = st.sidebar.number_input('Average Monthly Income')
        total_exp_per_month = st.sidebar.number_input('Monthly Expenditure')
        age_of_respondent = st.sidebar.number_input('Age of Respondent')
        chronic_illness = st.sidebar.checkbox('Chronic Illness')
        nhif_usage = st.sidebar.checkbox('Use NHIF')
        nssf_usage = st.sidebar.checkbox('Use NSSF')
        hse_land_loan = st.sidebar.checkbox('Use Loan to Purchase House/Land')
        securities_use = st.sidebar.checkbox('Invest in securities(Treasury Bill, MAkiba, etc)')
        land_house_ownership = st.sidebar.checkbox('Owns Land and/or House')
        electronic_device = st.sidebar.checkbox('Owns electronic device/s')
        motorvehicle_ownership = st.sidebar.checkbox('Owns motorvehicle/s')
        livestock_ownership = st.sidebar.checkbox('Owns livestock')
        insurance = st.sidebar.checkbox('Use Insurance')
         
        data = {'area': area,
                'age_of_respondent': age_of_respondent,
                'avg_mnth_income': avg_mnth_income,
                'total_exp_per_month': total_exp_per_month,
                'chronic_illness': chronic_illness,
                'nhif_usage': nhif_usage,
                'nssf_usage': nssf_usage,
                'income_source': income_source,
                'most_important_life_goal': most_important_life_goal,
                'hse_land_loan': hse_land_loan,
                'nearest_financial_prod': nearest_financial_prod,
                'securities_use': securities_use,
                'land_house_ownership': land_house_ownership,
                'electronic_device': electronic_device,
                'motorvehicle_ownership': motorvehicle_ownership,
                'livestock_ownership': motorvehicle_ownership,
                'insurance': insurance,}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

