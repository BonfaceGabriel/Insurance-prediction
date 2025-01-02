import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import lightgbm
import shap
from sklearn.preprocessing import OneHotEncoder


#data
data = pd.read_csv('App/data/Nakuru_FinAccess2.csv')
# st.write(data)

#load models
classifier = joblib.load('App/models/classifier.joblib')
cluster = joblib.load('App/models/new_model.joblib')
scaler = joblib.load('App/models/scaler.joblib')
encoder = joblib.load('App/models/encoder.joblib')

print(cluster.cost_)
#recommendended insurance products
def recommend_insurance(row):
    cluster_based_products = []
    # for index, row in df.iterrows():
    if (row['avg_mnth_income'] > 50000):  # High income cluster
        cluster_based_products.extend(['Life Insurance', 'Investment-linked Insurance'])
    elif  15000 < row['avg_mnth_income'] <= 50000:  # Medium income cluster
        cluster_based_products.append('Life Insurance')
    elif (row['avg_mnth_income'] <= 15000):  # Low income cluster
        cluster_based_products.append('Basic Insurance Package')

    # Add specific product recommendations based on characteristics
    if row['chronic_illness'] or row['nhif_usage'] or row['most_important_life_goal'] == 'Health':
        cluster_based_products.append("Health Insurance")
    if row['motorvehicle_ownership']:
        cluster_based_products.append("Motor Insurance")
    if row['land_house_ownership']  or (row['most_important_life_goal'] in ['Assets' or 'Home']):
        cluster_based_products.append("Property Insurance")
    if (row['most_important_life_goal'] and row['income_source']) == 'Business' :
        cluster_based_products.append("Business Insurance")
    if (row['most_important_life_goal'] == 'Education' ):
        cluster_based_products.append("Education Insurance")
    if row['livestock_ownership']:
        cluster_based_products.append("Livestock Insurance")
    if row['income_source'] == 'Agriculture':
        cluster_based_products.append("Agriculture Insurance")
    if row['age_of_respondent'] > 50 or row['nssf_usage'] or row['income_source'] == 'Pension':
        cluster_based_products.append("Retirement Insurance")

    return list(set(cluster_based_products))

#title
st.title('Insurance Product Prediction App')

#sidebar
st.sidebar.title('User Input Features')

container_style = """
    <style>
        .container1 {
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
            border-color: 'white';
        }
    </style>
  """

#main page content
c = st.container(border=True)
c.markdown('''
    This app provides clustering results and insurance product recommendations based on customer characteristics in Nakuru County.
    
    Features:
    - :grey-background[Cluster Prediction] based on user inputs
    - :grey-background[Insurance product recommendation] based on the cluster

    ''')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        income_source = st.sidebar.selectbox('Source of Income',('Casual_work','Family','Agriculture', 'Business', 'Employment', 'Renting', 'Pension', 'Aid'))
        most_important_life_goal = st.sidebar.selectbox('Most Important Life Goal',('Food','Education', 'Health', 'Business', 'Career', 'Home', 'Assets', 'None'))
        nearest_financial_prod = st.sidebar.selectbox('Nearest Financial Product',('MMoney','Bank', 'Insurance'))
        area = st.sidebar.selectbox('Area',('Urban','Rural'))
        avg_mnth_income = st.sidebar.number_input('Average Monthly Income', min_value=100)
        total_exp_per_month = st.sidebar.number_input('Monthly Expenditure', min_value = 500)
        age_of_respondent = st.sidebar.number_input('Age of Respondent', min_value=16)
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
st.subheader('User Input')

if uploaded_file:
    st.write(input_df.head())
else:
    st.write(input_df)

preprocess_data = input_df.copy()

#preprocess input data
#numerical column
num_cols = ['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month']
scaled_data = scaler.transform(preprocess_data[num_cols])
preprocess_data[num_cols] = scaled_data

#categorical column
cat_cols = ['most_important_life_goal', 'area', 'income_source', 'nearest_financial_prod']

encoded = encoder.transform(preprocess_data[cat_cols])
one_hot_df = pd.DataFrame(encoded, 
                        columns=encoder.get_feature_names_out(cat_cols))

#processed data
processed_data = pd.concat([preprocess_data.drop(cat_cols, axis=1), one_hot_df], axis=1)
cols = processed_data.select_dtypes(include=['object']).columns
processed_data[cols] = processed_data[cols].astype('bool')
print(processed_data.info()) 

#predict
prediction = classifier.predict(processed_data)
input_df['Clusters'] = prediction
input_df['recommended_products'] = input_df.apply(recommend_insurance, axis=1) 
insurance_products = input_df['recommended_products'].values[0]




#cluster analysis
cluster_data = data[data['Clusters'] == prediction[0]]

if st.button('Make Prediction'):
    if uploaded_file:
        st.write(input_df[['Clusters', 'recommended_products']])
    else:
        st.subheader('Insurance Product Prediction')
        cluster_size = len(cluster_data)
        average_income = round(cluster_data['avg_mnth_income'].mean(), 0)
        average_age = round(cluster_data['age_of_respondent'].mean(), 0)

        # Display Card
        c = st.container()
        c.markdown(
            f"""
            <div style="
                background-color: rgba(26, 26,36, 1);
                border: 2px solid #e3e3e3;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                font-family: Arial, sans-serif;
            ">
                <h3 style="text-align: center; color: #333;">Customer Cluster Information</h3>
                <p style="text-align: left; font-size: 16px; color: #707070;">
                    <b>The customer belongs to cluster {prediction}.</b><br>
                    <b style="color: #707070;">------------------------------------------------------</b><br>
                    <b>Size:</b> <em style="color: #707070;">{cluster_size} customers </em><br>
                    -------------------------------------------------------<br>
                    <b>RECOMMENDED INSURANCE PRODUCTS:</b><br>
                    {' '.join(f'<li style="color: #707070;"><em style="color: #707070;">{i}</em></li>' for i in insurance_products)}
                    <b style="color: #707070;">-------------------------------------------------------</b><br>
                    <b style="color: #707070;">Average Cluster Income:</b> <em style="color: #707070;">KES {average_income:,}</em><br>
                    <b style="color: #707070;">-------------------------------------------------------</b><br>
                    <b style="color: #707070;">Average Cluster Age:</b> <em style="color: #707070;"> {average_age} years</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('''
                ---
                Created with ❤️ by :red-background[Bonface Odhiambo]
                ''')



        





