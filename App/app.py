import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import lightgbm
from anthropic import Anthropic
import json 
from sklearn.preprocessing import OneHotEncoder
from ydata_profiling import ProfileReport
from bs4 import BeautifulSoup
from streamlit_card import card
import hydralit_components as hc
import time


st.set_page_config(
        page_title="Insurance Recommendation System",
        page_icon="🏥",
        layout="wide", 
        initial_sidebar_state="collapsed"
    )



st.markdown("""
        <style>
        .recommendation-card {
            background-color: rgba(49, 51, 63, 0.8);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .recommendation-header {
            color: #ffffff;
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        .recommendation-content {
            color: #e6e6e6;
            font-size: 1em;
            line-height: 1.5;
        }
        .info-box {
            background-color: rgba(32, 33, 36, 0.8);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

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

#data
data = pd.read_csv('./data/Nakuru_FinAccess2.csv')


#load models
classifier = joblib.load('./models/classifier.joblib')
cluster = joblib.load('./models/new_model.joblib')
scaler = joblib.load('./models/scaler.joblib')
encoder = joblib.load('./models/encoder.joblib')

def preprocess(df):
    num_cols = ['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month']
    cat_cols = ['most_important_life_goal', 'area', 'income_source', 'nearest_financial_prod']
    scaled_data = scaler.transform(df[num_cols])
    df[num_cols] = scaled_data
    encoded = encoder.transform(df[cat_cols])
    one_hot_df = pd.DataFrame(encoded, 
                        columns=encoder.get_feature_names_out(cat_cols))
    processed_data = pd.concat([df.drop(cat_cols, axis=1), one_hot_df], axis=1)
    cols = processed_data.select_dtypes(include=['object']).columns
    processed_data[cols] = processed_data[cols].astype('bool')
    return processed_data


class InsuranceRecommender:
    def __init__(self, model_path, api_key):
        with open(model_path, "rb") as model_file:
            self.cluster_model = joblib.load(model_file)
        self.client = Anthropic(api_key=api_key)

    def call_claude_api(self, prompt):
        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    def extract_report_content(self, profile_report):
        try:
            # Initialize default values
            text_summary = ""
            numeric_distributions = {}
            correlation_matrix = {}

            # Extract text summary
            if profile_report is None:
                raise ValueError("Profile report is None")
                
            profile_html = profile_report.to_html()
            soup = BeautifulSoup(profile_html, "html.parser")
            text_summary = " ".join([el.get_text(strip=True) for el in soup.find_all(["h1", "h2", "p", "li", "td", "span"])])

            #get correlation matrix
            try:
                correlation_matrix = profile_report.description_set.correlations.pearson
                if correlation_matrix is None:
                    correlation_matrix = {}
            except AttributeError:
                try:
                    correlation_matrix = profile_report.report.correlations.pearson
                    if correlation_matrix is None:
                        correlation_matrix = {}
                except AttributeError:
                    correlation_matrix = {}

            #get numeric distributions
            try:
                variables = profile_report.description_set.variables
                for col, var_data in variables.items():
                    if hasattr(var_data, 'type') and var_data.type == "Numeric":
                        numeric_distributions[col] = {
                            "mean": getattr(var_data, 'mean', 0),
                            "std": getattr(var_data, 'std', 0),
                            "min": getattr(var_data, 'min', 0),
                            "max": getattr(var_data, 'max', 0),
                            "histogram": getattr(var_data, 'histogram', {}).get('counts', [])
                        }
            except AttributeError:
                    pass

            return text_summary, numeric_distributions, correlation_matrix

        except Exception as e:
            print(f"Error in extract_report_content: {str(e)}")
            # Return default values if something goes wrong
            return "", {}, {}



    def generate_recommendations(self, text_summary, numeric_distributions, correlation_matrix):
        prompt = f"""Given the following profile report: {text_summary}\n
        Distributions: {json.dumps(numeric_distributions)}\n 
        and Correlations: {json.dumps(correlation_matrix)}\n 
        suggest top 5 insurance products that the customers belonging to this cluster would get 
        and give comprehensive reasonings for each.
        The format should be:
            The opening statements should read:Recommended Products based on Cluster Profile similar to yours:

            1. Product 
                - List of reasons why this product is recommended: 
        Remove the usual beginning and ending phrases of a prompt, 
        and the last part should be the last recommended product and 
        its list of reasons. In this formatting,
        assume this tool is going to be used by a sales agent to recommend products to a customer.
        """
        result = self.call_claude_api(prompt) 
        return result
    def analyze_customer_inputs(self, customer_data, cluster_recommendations):
        prompt = f"""Given the following Customer Data: {customer_data}\n 
        and the Cluster Recommendations: {cluster_recommendations} 
        for the cluster the customer belongs to, \n
        analyze the customer attributes 
        and suggest products unique to the customer based on their input which are not included in {cluster_recommendations}. 
        Title them 'Products Unique to the Customer'. 
        The format should be:
            Products Unique to the Customer:

            1. Product 
                - List of reasons why this product is recommended

        Remove the usual beginning and ending phrases of a prompt, 
        and the last part should be the last recommended product and 
        its list of reasons. In this formatting,
         assume this tool is going to be used by a sales agent to recommend products to a customer.
        """
        result = self.call_claude_api(prompt)
        return result
    def get_customer_recommendations(self, preprocessed_data, customer_data):
        cluster = self.cluster_model.predict(preprocessed_data)[0]
        cluster_data = data[data['Clusters'] == cluster]
        profile_report = ProfileReport(cluster_data, minimal=True)

        text_summary, numeric_distributions, correlation_matrix = self.extract_report_content(profile_report)
        cluster_recommendations = self.generate_recommendations(text_summary, numeric_distributions, correlation_matrix)
        customer_specific_recommendations = self.analyze_customer_inputs(customer_data, cluster_recommendations)

        return {
            "Predicted Cluster": cluster,
            "Cluster Recommendations": cluster_recommendations,
            "Products Unique to the Customer": customer_specific_recommendations
        }

recommender = InsuranceRecommender(model_path="./models/classifier.joblib", api_key=st.secrets["api_key"])



#title
st.title('AI-Powered Insurance Product Recommendation System')

#sidebar
st.sidebar.title('User Input Features')

#main page content
c = st.container(border=True)
c.markdown('''
    This app provides clustering results and insurance product recommendations based on customer characteristics in Nakuru County.
    
    Features:
    - :grey-background[Cluster Prediction] based on user inputs
    - :grey-background[Insurance product recommendation] based on the cluster

    ''')


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

preprocess_data = input_df.copy()
preprocess_data = preprocess(preprocess_data)



if st.button('Get Recommendations'):
    with st.spinner('Getting Recommendations...'):
        # Initialize the recommender
        recommender = InsuranceRecommender(
            model_path="./models/classifier.joblib",
            api_key=st.secrets["api_key"]
        )

    
        # Get recommendations
        recommendations = recommender.get_customer_recommendations(preprocessed_data=preprocess_data, customer_data=input_df)

        # Display recommendations in expandable sections with custom styling
        st.markdown("## Insurance Recommendations")
        
        #Cluster
        with st.expander("👥 Customer Cluster", expanded=True):
            st.markdown(f"The customer belongs to Cluster {recommendations['Predicted Cluster']}")

        # Cluster recommendations
        with st.expander("📊 Cluster-Based Recommendations", expanded=True):
            st.markdown("""
                <div style='background-color: #707070; padding: 20px; border-radius: 10px;'>
                    <h4>Based on profiles similar to yours</h4>
                    {}
                </div>
            """.format(recommendations["Cluster Recommendations"]), unsafe_allow_html=True)
            

        # Personal recommendations
        with st.expander("👤 Personalized Recommendations", expanded=True):
            st.markdown("""
                <div style='background-color: #707070; padding: 20px; border-radius: 10px;'>
                    <h4>Tailored specifically for you</h4>
                    {}
                </div>
            """.format(recommendations["Products Unique to the Customer"]), unsafe_allow_html=True)
        
        


st.markdown('''
            ---
            Created with ❤️ by :red-background[Bonface Odhiambo]
            ''')



        





