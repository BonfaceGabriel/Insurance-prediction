import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import lightgbm
import shap
from sklearn.preprocessing import StandardScaler


#data
data = pd.read_csv('App/data/Nakuru_FinAccess.csv')
data.drop(columns=['HHNo', 'income_bins'], inplace=True)
data['most_important_life_goal'] = data['most_important_life_goal'].fillna('None')
cluster1 = data[data['Clusters'] == 0]
cluster2 = data[data['Clusters'] == 1]
cluster3 = data[data['Clusters'] == 2]

#SHAP
shap_data = data.copy()
shap_data = shap_data.drop(columns='Clusters')

#cluster_data
cluster_data = shap_data.copy()

#preprocess data
#scale numerical columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(shap_data[['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month']])
shap_data[['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month']] = scaled_data

#encode categorical columns
cat_cols = ['most_important_life_goal', 'area', 'income_source', 'nearest_financial_prod']
encoded = pd.get_dummies(shap_data[cat_cols])

#join the data
shap_data = pd.concat([shap_data.drop(cat_cols, axis=1), encoded], axis=1)

#convert object type columns to bool for shap model
cols = shap_data.select_dtypes(include=['object']).columns
shap_data[cols] = shap_data[cols].astype('bool')


#load models
classifier = joblib.load('App/models/classifier.joblib')
cluster = joblib.load('App/models/kproto.joblib')
 
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
c = st.container(border=True)
c.markdown('''
    This app provides clustering results and insurance product recommendations based on customer characteristics in Nakuru County.
    
    Features:
    - :grey-background[SHAP summary plots] to show cluster characteristics
    - :grey-background[Cluster Prediction] based on user inputs
    - :grey-background[Insurance product recommendation] based on the cluster

    ''')

if st.sidebar.checkbox("Show SHAP Summary Plots"):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(shap_data)

            c1, c2 = st.columns(2)
            with c1:
                st.write("SHAP Summary for Cluster 1")
                shap_values_for_cluster1 = shap_values[:, :, 0]
                fig, ax = plt.subplots()
                ax = shap.summary_plot(shap_values_for_cluster1, shap_data, plot_type="bar", show=False)
                st.pyplot(fig)
            with c2:
                st.write("SHAP Summary for Cluster 2")
                shap_values_for_cluster2 = shap_values[:, :, 1]
                fig, ax = plt.subplots()
                ax = shap.summary_plot(shap_values_for_cluster2, shap_data, plot_type="bar", show=False)
                st.pyplot(fig)

            st.write("SHAP Summary for Cluster 3")
            shap_values_for_cluster3 = shap_values[:, :, 2]
            fig, ax = plt.subplots()
            ax = shap.summary_plot(shap_values_for_cluster3, shap_data, plot_type="bar", show=False)
            st.pyplot(fig)

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
    st.write(input_df)
    
    #prediction data
    pred_data = pd.concat([cluster_data, input_df]).reset_index(drop=True)
    

    #preprocess input data
    #numerical column
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pred_data[['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month']])
    pred_data[['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month']] = scaled_data

    #categorical columns
    cat_cols = ['most_important_life_goal', 'area', 'income_source', 'nearest_financial_prod']
    encoded = pd.get_dummies(pred_data[cat_cols])

    #processed data
    processed_data = pd.concat([pred_data.drop(cat_cols, axis=1), encoded], axis=1)
    cols = processed_data.select_dtypes(include=['object']).columns
    processed_data[cols] = processed_data[cols].astype('bool')
   

    #predict
    pred_df = processed_data.tail(1)
    prediction = classifier.predict(pred_df)


    if st.button('Make Prediction'):
        st.subheader('Insurance Product Prediction')
        if prediction == 0:
            c = st.container(border=True)
            c.markdown('''
                The customer belongs to cluster 1.

                Cluster Description:
                - Old low income farmers
                
                Probable insurance product:
                - Crop Insurance
                - Livestock Insurance
                - Basic Health Insurance
                - Term Life Insurance

                ''')
        elif prediction == 1:
            c = st.container(border=True)
            c.markdown('''
                The customer belongs to Cluster 2.
                
                Cluster Description:
                - Young small business owners
                
                Probable insurance product:
                - Business property insurance
                - General liability insurance
                - Professional liability insurance(if applicable to the business)

                ''')
        else:
            c = st.container(border=True)
            c.markdown('''
                The customer belongs to Cluster 3.

                Cluster Description:
                - Middle aged, average sized business owners 
                
                Probable insurance product:
                - Workers compensation Insurance(if applicable)
                - Business property insurance
                - Product liability insurance
                - Life insurance

                ''')

    st.markdown('''
                ---
                Created with ❤️ by :red-background[Bonface Odhiambo]
                ''')



        

    



