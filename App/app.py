import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from anthropic import Anthropic
import json 
from sklearn.preprocessing import OneHotEncoder
from ydata_profiling import ProfileReport
from bs4 import BeautifulSoup
from typing import Dict
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler
from google import genai


st.set_page_config(
    page_title="InsureAI | Smart Insurance Recommendations",
    page_icon="🛡️",
        layout="wide", 
        initial_sidebar_state="collapsed"
    )



st.markdown("""
        <style>
    .main {
        padding: 0rem 5rem;
    }
    .title-container {
        padding: 1.5rem 0rem;
        margin-bottom: 3rem;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5E88 100%);
            border-radius: 10px;
    }
    .title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        padding: 1rem;
    }
    .subtitle {
        color: #E0E0E0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0;
        padding: 0.5rem;
    }
    .feature-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .input-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
            border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Remove white background from tab buttons */
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: white !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Style for customer info header */
    .customer-info-header {
        color: white;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Remove white background from input section */
    .input-section {
        background-color: transparent !important;
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# container_style = """
#     <style>
#         .container1 {
#             border: 2px solid #3498db;
#             border-radius: 8px;
#             padding: 10px;
#             margin-bottom: 20px;
#             border-color: 'white';
#         }
#     </style>
#   """

st.markdown("""
    <style>
    .main {
        padding: 0rem 5rem;
    }
    .title-container {
        padding: 1.5rem 0rem;
        margin-bottom: 3rem;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5E88 100%);
        border-radius: 10px;
    }
    .title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        padding: 1rem;
    }
    .subtitle {
        color: #E0E0E0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 400;
        margin: 0;
        padding: 0.5rem;
    }
    .feature-card {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .input-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
            border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)

#data
data = pd.read_csv('App/data/nakuru_dataset.csv')
data['insurance'] = data['insurance'].map({'Yes': True, 'No': False})



#load models
classifier = joblib.load('App/models/classifier.joblib')
# cluster = joblib.load('App/models/new_model.joblib')
scaler = joblib.load('App/models/scaler.joblib')
encoder = joblib.load('App/models/encoder.joblib')

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

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class InsuranceProducts:
    """Defines available insurance products and their coverage"""
    
    def __init__(self):
        self.products = {
            'HEALTH': {
                'PREMIUM': {
                    'name': 'Comprehensive Health Cover',
                    'coverage': 1000000,
                    'benefits': [
                        'Inpatient & Outpatient Care',
                        'Chronic Disease Management',
                        'Dental & Optical',
                        'Maternity Cover',
                        'International Treatment',
                        'Mental Health Coverage',
                        'Alternative Medicine'
                    ],
                    'waiting_period': '30 days',
                    'min_premium': 5000,
                    'payment_plans': {
                        'monthly': {'amount': 5000, 'discount': 0},
                        'quarterly': {'amount': 14500, 'discount': 3},
                        'annually': {'amount': 55000, 'discount': 8}
                    }
                },
                'STANDARD': {
                    'name': 'Standard Health Cover',
                    'coverage': 500000,
                    'benefits': [
                        'Inpatient & Outpatient Care',
                        'Limited Chronic Disease Management',
                        'Basic Dental & Optical',
                        'Basic Maternity Cover'
                    ],
                    'waiting_period': '60 days',
                    'min_premium': 2500,
                    'payment_plans': {
                        'monthly': {'amount': 2500, 'discount': 0},
                        'quarterly': {'amount': 7200, 'discount': 4},
                        'annually': {'amount': 27000, 'discount': 10}
                    }
                },
                'BASIC': {
                    'name': 'Basic Health Cover',
                    'coverage': 200000,
                    'benefits': [
                        'Inpatient Care',
                        'Limited Outpatient Care',
                        'Emergency Services'
                    ],
                    'waiting_period': '90 days',
                    'min_premium': 1000,
                    'payment_plans': {
                        'monthly': {'amount': 1000, 'discount': 0},
                        'quarterly': {'amount': 2900, 'discount': 3},
                        'annually': {'amount': 11000, 'discount': 8}
                    }
                }
            },
            'LIFE': {
                'PREMIUM': {
                    'name': 'Comprehensive Life Insurance',
                    'coverage': 5000000,
                    'benefits': [
                        'Death Benefit',
                        'Critical Illness Cover',
                        'Disability Cover',
                        'Investment Component',
                        'Education Benefit',
                        'Retirement Planning',
                        'Estate Planning'
                    ],
                    'waiting_period': 'None',
                    'min_premium': 7500,
                    'payment_plans': {
                        'monthly': {'amount': 7500, 'discount': 0},
                        'quarterly': {'amount': 21750, 'discount': 3},
                        'annually': {'amount': 82500, 'discount': 8}
                    }
                },
                'STANDARD': {
                    'name': 'Standard Life Insurance',
                    'coverage': 2000000,
                    'benefits': [
                        'Death Benefit',
                        'Critical Illness Cover',
                        'Disability Cover',
                        'Basic Investment Component'
                    ],
                    'waiting_period': 'None',
                    'min_premium': 3500,
                    'payment_plans': {
                        'monthly': {'amount': 3500, 'discount': 0},
                        'quarterly': {'amount': 10150, 'discount': 3},
                        'annually': {'amount': 38500, 'discount': 8}
                    }
                }
            },
            'PROPERTY': {
                'PREMIUM': {
                    'name': 'Comprehensive Property Insurance',
                    'coverage': 10000000,
                    'benefits': [
                        'Building Coverage',
                        'Contents Coverage',
                        'Natural Disasters',
                        'Theft & Burglary',
                        'Liability Coverage',
                        'Business Interruption',
                        'Rental Income Protection'
                    ],
                    'waiting_period': '14 days',
                    'min_premium': 10000,
                    'payment_plans': {
                        'monthly': {'amount': 10000, 'discount': 0},
                        'quarterly': {'amount': 29000, 'discount': 3},
                        'annually': {'amount': 110000, 'discount': 8}
                    }
                }
            },
            'BUSINESS': {
                'PREMIUM': {
                    'name': 'Business Insurance Plus',
                    'coverage': 15000000,
                    'benefits': [
                        'Property Coverage',
                        'Business Interruption',
                        'Public Liability',
                        'Professional Indemnity',
                        'Cyber Security Coverage',
                        'Employee Coverage'
                    ],
                    'waiting_period': '30 days',
                    'min_premium': 15000,
                    'payment_plans': {
                        'monthly': {'amount': 15000, 'discount': 0},
                        'quarterly': {'amount': 43500, 'discount': 3},
                        'annually': {'amount': 165000, 'discount': 8}
                    }
                }
            },
            'AGRICULTURE': {
                'PREMIUM': {
                    'name': 'Agricultural Insurance',
                    'coverage': 5000000,
                    'benefits': [
                        'Crop Insurance',
                        'Livestock Coverage',
                        'Equipment Protection',
                        'Weather Index Insurance',
                        'Storage Facility Coverage'
                    ],
                    'waiting_period': '30 days',
                    'min_premium': 5000,
                    'payment_plans': {
                        'monthly': {'amount': 5000, 'discount': 0},
                        'quarterly': {'amount': 14500, 'discount': 3},
                        'annually': {'amount': 55000, 'discount': 8}
                    }
                }
            },
            'EDUCATION': {
                'PREMIUM': {
                    'name': 'Education Savings Plan',
                    'coverage': 3000000,
                    'benefits': [
                        'Guaranteed Education Fund',
                        'Investment Returns',
                        'Life Insurance Component',
                        'Flexible Payment Options',
                        'University Fee Protection'
                    ],
                    'waiting_period': 'None',
                    'min_premium': 3000,
                    'payment_plans': {
                        'monthly': {'amount': 3000, 'discount': 0},
                        'quarterly': {'amount': 8700, 'discount': 3},
                        'annually': {'amount': 33000, 'discount': 8}
                    }
                }
            }
        }

class RiskAssessment:
    """Handles customer risk calculation"""
    def __init__(self):
        self.risk_weights = {
            'age': 0.15,
            'income': 0.20,
            'health': 0.25,
            'assets': 0.20,
            'financial_behavior': 0.20
        }
        
    def calculate_risk_score(self, customer_data: pd.DataFrame) -> Dict:
        # Age risk
        age = customer_data['age_of_respondent'].values[0]
        age_risk = self._calculate_age_risk(age)
        
        # Income risk
        income = customer_data['avg_mnth_income'].values[0]
        expenses = customer_data['total_exp_per_month'].values[0]
        income_risk = self._calculate_income_risk(income, expenses)
        
        # Health risk
        health_risk = self._calculate_health_risk(
            customer_data['chronic_illness'].values[0],
            customer_data['nhif_usage'].values[0]
        )
        
        # Asset risk
        asset_risk = self._calculate_asset_risk(customer_data)
        
        # Financial behavior risk
        financial_risk = self._calculate_financial_behavior_risk(customer_data)
        
        # Calculate weighted risk score
        total_risk = (
            age_risk * self.risk_weights['age'] +
            income_risk * self.risk_weights['income'] +
            health_risk * self.risk_weights['health'] +
            asset_risk * self.risk_weights['assets'] +
            financial_risk * self.risk_weights['financial_behavior']
        )
        
        return {
            'total_risk': total_risk,
            'components': {
                'age_risk': age_risk,
                'income_risk': income_risk,
                'health_risk': health_risk,
                'asset_risk': asset_risk,
                'financial_risk': financial_risk
            }
        }
    
    def _calculate_age_risk(self, age: int) -> float:
        if age < 25:
            return 0.7  # Higher risk for very young
        elif age < 35:
            return 0.4
        elif age < 50:
            return 0.5
        else:
            return 0.8  # Higher risk for older
            
    def _calculate_income_risk(self, income: float, expenses: float) -> float:
        savings_ratio = (income - expenses) / income if income > 0 else 0
        if savings_ratio < 0:
            return 1.0
        elif savings_ratio < 0.2:
            return 0.8
        elif savings_ratio < 0.4:
            return 0.5
        else:
            return 0.3
            
    def _calculate_health_risk(self, has_chronic_illness: bool, has_nhif: bool) -> float:
        base_risk = 0.8 if has_chronic_illness else 0.4
        return base_risk * (0.7 if not has_nhif else 1.0)
        
    def _calculate_asset_risk(self, data: pd.DataFrame) -> float:
        risk_score = 0.5  # Base risk
        
        # Reduce risk if customer owns assets
        if data['land_house_ownership'].values[0]:
            risk_score -= 0.1
        if data['motorvehicle_ownership'].values[0]:
            risk_score -= 0.1
        if data['electronic_device'].values[0]:
            risk_score -= 0.05
            
        return max(0.2, risk_score)  # Minimum risk of 0.2
        
    def _calculate_financial_behavior_risk(self, data: pd.DataFrame) -> float:
        risk_score = 0.7  # Base risk
        
        # Reduce risk for positive financial behaviors
        if data['nssf_usage'].values[0]:
            risk_score -= 0.1
        if data['securities_use'].values[0]:
            risk_score -= 0.1
        if data['insurance'].values[0]:
            risk_score -= 0.1
            
        return max(0.3, risk_score)  # Minimum risk of 0.3

class CustomerProfile:
    """Handles customer profiling and segmentation"""
    def __init__(self):
        self.segments = {
            'PREMIUM': {'min_income': 100000, 'min_assets': 2},
            'STANDARD': {'min_income': 50000, 'min_assets': 1},
            'BASIC': {'min_income': 0, 'min_assets': 0}
        }
        
    def create_profile(self, customer_data: pd.DataFrame) -> Dict:
        # Count assets
        asset_count = sum([
            customer_data['land_house_ownership'].values[0],
            customer_data['motorvehicle_ownership'].values[0],
            customer_data['electronic_device'].values[0]
        ])
        
        # Determine segment
        income = customer_data['avg_mnth_income'].values[0]
        segment = self._determine_segment(income, asset_count)
        
        return {
            'segment': segment,
            'profile': {
                'age': customer_data['age_of_respondent'].values[0],
                'income': income,
                'monthly_expenses': customer_data['total_exp_per_month'].values[0],
                'area': customer_data['area'].values[0],
                'income_source': customer_data['income_source'].values[0],
                'life_goal': customer_data['most_important_life_goal'].values[0],
                'asset_count': asset_count,
                'has_insurance': customer_data['insurance'].values[0],
                'has_chronic_illness': customer_data['chronic_illness'].values[0]
            }
        }
        
    def _determine_segment(self, income: float, asset_count: int) -> str:
        for segment, criteria in self.segments.items():
            if (income >= criteria['min_income'] and 
                asset_count >= criteria['min_assets']):
                return segment
        return 'BASIC'

class InsuranceRecommender:
    def __init__(self, model_path, api_key):
        with open(model_path, "rb") as model_file:
            self.cluster_model = joblib.load(model_file)
        self.client = Anthropic(api_key=api_key)
        # self.client = genai.Client(api_key = api_key)
        self.risk_assessor = RiskAssessment()
        self.profiler = CustomerProfile()
        self.products = InsuranceProducts()
        
        # Create background data for SHAP
        try:
            background_data = shap.sample(data, 100)
            # LightGBM has built-in SHAP support
            self.explainer = shap.TreeExplainer(self.cluster_model)
        except Exception as e:
            print(f"Error initializing SHAP explainer: {str(e)}")
            self.explainer = None

    def call_claude_api(self, prompt):
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    # def call_claude_api(self, prompt):
    #     response = self.client.models.generate_content(
    #         model="gemini-2.5-pro",
    #         contents=prompt
           
    #     )
    #     return response.text
    

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

    def get_feature_importance(self, preprocessed_data):
        """Calculate SHAP values for the customer data"""
        try:
            if self.explainer is None:
                return {
                    'importance_dict': {},
                    'sorted_features': [],
                    'shap_values': None
                }
                
            shap_values = self.explainer.shap_values(preprocessed_data)
            feature_names = preprocessed_data.columns.tolist()
            
            importance_dict = {}
            
            if isinstance(shap_values, list):
                mean_shap_values = np.abs(np.mean(shap_values, axis=0))
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = float(np.mean(np.abs(mean_shap_values[:, i])))
            else:
                for i, feature in enumerate(feature_names):
                    importance_dict[feature] = float(np.mean(np.abs(shap_values[:, i])))
            
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
            
            return {
                'importance_dict': importance_dict,
                'sorted_features': sorted_features,
                'shap_values': shap_values
            }
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return {
                'importance_dict': {},
                'sorted_features': [],
                'shap_values': None
            }

    def generate_recommendations(self, text_summary, numeric_distributions, correlation_matrix, 
                           risk_assessment, customer_profile, available_products, 
                           feature_importance):
        try:
            # Prepare safe JSON-serializable data
            risk_assessment_safe = {
                'total_risk': float(risk_assessment['total_risk']),
                'components': {k: float(v) for k, v in risk_assessment['components'].items()}
            }
            
            profile_safe = {}
            for k, v in customer_profile['profile'].items():
                if isinstance(v, (np.integer, np.floating)):
                    profile_safe[k] = float(v)
                elif isinstance(v, np.bool_):
                    profile_safe[k] = bool(v)
                else:
                    profile_safe[k] = v

            feature_importance_safe = {
                'sorted_features': [(str(k), float(v)) for k, v in feature_importance['sorted_features'][:5]]
            }

            # First prompt for cluster-based recommendations
            cluster_prompt = f"""Given the following cluster information:

            1. Profile Report: {text_summary}
            2. Statistical Distributions: {json.dumps(numeric_distributions, cls=NumpyEncoder)}
            3. Correlations: {json.dumps(correlation_matrix, cls=NumpyEncoder)}
            4. Available Products: {json.dumps(available_products)}

            Provide insurance recommendations for this customer cluster.
            Focus on products that would benefit the majority of customers in this cluster.
            Include payment plans and explain why each product suits the cluster profile.

            Format as:
            Cluster-Based Recommendations:

            1. [Product Name] - [Tier Level]
            - Coverage: [Amount]
            - Payment Plans: [List payment options with discounts]
            - Key Benefits: [List]
            - Cluster Fit: [Why this product suits this customer segment]
            """
            
            cluster_recommendations = self.call_claude_api(cluster_prompt)

            # Second prompt for personalized recommendations
            personal_prompt = f"""Given the following customer-specific information:

            1. Risk Assessment:
            - Overall Risk Score: {risk_assessment_safe['total_risk']}
            - Risk Components: {json.dumps(risk_assessment_safe['components'])}
            2. Customer Profile:
            - Segment: {customer_profile['segment']}
            - Details: {json.dumps(profile_safe)}
            3. Key Influential Features:
            {json.dumps(dict(feature_importance_safe['sorted_features']))}
            4. Available Products: {json.dumps(available_products)}
            5. Existing Cluster Recommendations: {cluster_recommendations}

            Provide personalized insurance recommendations that are different from the cluster recommendations.
            Focus on the customer's unique characteristics, risk profile, and financial capacity.
            Include payment plans and explain why each product suits their specific situation.

            Format as:
            Personalized Recommendations:

            1. [Product Name] - [Tier Level]
            - Coverage: [Amount]
            - Payment Plans: [List payment options with discounts]
            - Key Benefits: [List]
            - Personal Fit: [Why this product suits this specific customer]
            """

            personal_recommendations = self.call_claude_api(personal_prompt)

            return {
                "cluster_recommendations": cluster_recommendations,
                "personal_recommendations": personal_recommendations
            }

        except Exception as e:
            print(f"Error in generate_recommendations: {str(e)}")
            return {
                "cluster_recommendations": "Error generating cluster recommendations.",
                "personal_recommendations": "Error generating personal recommendations."
            }

    def get_customer_recommendations(self, preprocessed_data, customer_data):
        # Get cluster and profile data
        cluster = self.cluster_model.predict(preprocessed_data)[0]
        cluster_data = data[data['Clusters'] == cluster]
        profile_report = ProfileReport(cluster_data, minimal=True)

        # Calculate feature importance using SHAP
        feature_importance = self.get_feature_importance(preprocessed_data)

        # Calculate risk assessment
        risk_assessment = self.risk_assessor.calculate_risk_score(customer_data)
        
        # Create customer profile
        customer_profile = self.profiler.create_profile(customer_data)
        
        # Get available products
        available_products = self.products.products

        # Generate recommendations
        text_summary, numeric_distributions, correlation_matrix = self.extract_report_content(profile_report)
        recommendations = self.generate_recommendations(
            text_summary, 
            numeric_distributions, 
            correlation_matrix,
            risk_assessment,
            customer_profile,
            available_products,
            feature_importance
        )

        return {
            "Predicted Cluster": cluster,
            "Risk Assessment": risk_assessment,
            "Customer Profile": customer_profile,
            "Recommendations": recommendations,
            "Cluster Data": cluster_data,
            "Feature Importance": feature_importance
        }

recommender = InsuranceRecommender(model_path="App/models/classifier.joblib", api_key=st.secrets["api_key"])



#title
st.title('AI-Powered Insurance Product Recommendation System')


#main page content
c = st.container(border=True)
c.markdown('''
    This app provides clustering results and insurance product recommendations based on customer characteristics in Nakuru County.
    
    Features:
    - :grey-background[Cluster Prediction] based on user inputs
    - :grey-background[Insurance product recommendation] based on the cluster

    ''')


def user_input_features():
    st.markdown("""
        <style>
        /* Remove white background from tab buttons */
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: white !important;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        
        
        /* Remove white background from input section */
        .input-section {
            background-color: transparent !important;
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 20px;
        }
        </style>
        
        # <h2 class='customer-info-header'>Customer Information</h2>
    """, unsafe_allow_html=True)
    st.subheader('Customer Information')
    
    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        # Personal Information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Personal Details")
            age_of_respondent = st.number_input('Age', min_value=16, max_value=100, value=30)
            area = st.selectbox('Area', ['Urban', 'Rural'])
            chronic_illness = st.checkbox('Has Chronic Illness')
            
        with col2:
            st.markdown("#### Financial Information")
            avg_mnth_income = st.number_input('Monthly Income (KES)', min_value=100, value=50000)
            total_exp_per_month = st.number_input('Monthly Expenses (KES)', min_value=500, value=30000)
            
        # Additional Information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Income & Goals")
            income_source = st.selectbox('Source of Income', 
                ['Casual_work', 'Family', 'Agriculture', 'Business', 
                'Employment', 'Renting', 'Pension', 'Aid'])
            most_important_life_goal = st.selectbox('Most Important Life Goal',
                ['Food', 'Education', 'Health', 'Business', 
                'Career', 'Home', 'Assets', 'None'])
                
        with col2:
            st.markdown("#### Financial Products")
            nearest_financial_prod = st.selectbox('Nearest Financial Product', 
                ['MMoney', 'Bank', 'Insurance'])
            nhif_usage = st.checkbox('Has NHIF Coverage')
            nssf_usage = st.checkbox('Has NSSF Coverage')
            
        with col3:
            st.markdown("#### Assets & Loans")
            hse_land_loan = st.checkbox('Has House/Land Loan')
            securities_use = st.checkbox('Invests in Securities')
            land_house_ownership = st.checkbox('Owns Land/House')
            electronic_device = st.checkbox('Owns Electronic Devices')
            motorvehicle_ownership = st.checkbox('Owns Vehicle')
            livestock_ownership = st.checkbox('Owns Livestock')
            insurance = st.checkbox('Has Insurance')

        st.markdown('</div>', unsafe_allow_html=True)
        
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
            'livestock_ownership': livestock_ownership,
            'insurance': insurance,}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
st.subheader('User Input')
st.write(input_df)

preprocess_data = input_df.copy()
preprocess_data = preprocess(preprocess_data)

def create_dashboard(recommendations):

    st.markdown("""
        <style>
        .recommendation-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #1E3D59;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tabs = st.tabs([
        "📊 Overview", 
        "🎯 Recommendations", 
        "⚠️ Risk Analysis", 
        "👥 Customer Profile",
        "📈 Cluster Insights"
    ])

    # Overview Tab
    with tabs[0]:
        st.markdown("### Customer Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_score = recommendations["Risk Assessment"]["total_risk"] * 100
            st.metric("Risk Score", f"{risk_score:.1f}%")
        with col2:
            segment = recommendations["Customer Profile"]["segment"]
            st.metric("Customer Segment", segment)
        with col3:
            cluster = recommendations["Predicted Cluster"]
            st.metric("Customer Cluster", f"Cluster {cluster}")

    # Recommendations Tab
    with tabs[1]:
        # Create two columns for cluster and personal recommendations
        cluster_col, personal_col = st.columns(2)
        
        with cluster_col:
            st.markdown("""
                <div style='background-color: rgba(49, 51, 63, 0.8); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: #ffffff;'>🔍 Cluster-Based Recommendations</h3>
                    <p style='color: #e6e6e6;'>Products recommended based on similar customer profiles</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Parse and display cluster recommendations
            cluster_recs = recommendations["Recommendations"]["cluster_recommendations"]
            for rec in cluster_recs.split("\n\n"):
                if rec.strip().startswith("1.") or rec.strip().startswith("2.") or rec.strip().startswith("3."):
                    with st.expander(rec.split("\n")[0].strip()):
                        # Create a card-like display for each recommendation
                        st.markdown("""
                            <div style='background-color: rgba(32, 33, 36, 0.8); padding: 15px; border-radius: 8px;'>
                            {}
                            </div>
                            """.format(rec.replace("\n", "<br>")), unsafe_allow_html=True)
        
        with personal_col:
            st.markdown("""
                <div style='background-color: rgba(49, 51, 63, 0.8); padding: 20px; border-radius: 10px;'>
                    <h3 style='color: #ffffff;'>👤 Personalized Recommendations</h3>
                    <p style='color: #e6e6e6;'>Products tailored to your specific needs</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Parse and display personal recommendations
            personal_recs = recommendations["Recommendations"]["personal_recommendations"]
            for rec in personal_recs.split("\n\n"):
                if rec.strip().startswith("1.") or rec.strip().startswith("2.") or rec.strip().startswith("3."):
                    with st.expander(rec.split("\n")[0].strip()):
                        # Create a card-like display for each recommendation
                        st.markdown("""
                            <div style='background-color: rgba(32, 33, 36, 0.8); padding: 15px; border-radius: 8px;'>
                            {}
                            </div>
                            """.format(rec.replace("\n", "<br>")), unsafe_allow_html=True)

    # Risk Analysis Tab
    with tabs[2]:
        st.markdown("### Risk Assessment Breakdown")
        
        # Risk components visualization
        risk_data = recommendations["Risk Assessment"]["components"]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Radar chart for risk components
            fig = go.Figure(data=go.Scatterpolar(
                r=[risk_data[k] * 100 for k in risk_data.keys()],
                theta=list(risk_data.keys()),
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Risk Component Analysis"
            )
            st.plotly_chart(fig)
        
        with col2:
            # Risk breakdown table
            st.markdown("#### Risk Components")
            for component, value in risk_data.items():
                st.metric(
                    component.replace('_', ' ').title(),
                    f"{value * 100:.1f}%"
                )

    # Customer Profile Tab
    with tabs[3]:
        profile = recommendations["Customer Profile"]["profile"]
        
        # Create three columns for different aspects of the profile
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Demographics")
            st.markdown(f"- **Age:** {profile['age']}")
            st.markdown(f"- **Area:** {profile['area']}")
            st.markdown(f"- **Life Goal:** {profile['life_goal']}")
        
        with col2:
            st.markdown("### Financial Status")
            st.markdown(f"- **Monthly Income:** KES {profile['income']:,.2f}")
            st.markdown(f"- **Monthly Expenses:** KES {profile['monthly_expenses']:,.2f}")
            savings_rate = (profile['income'] - profile['monthly_expenses']) / profile['income'] * 100
            st.markdown(f"- **Savings Rate:** {savings_rate:.1f}%")
        
        with col3:
            st.markdown("### Assets & Coverage")
            st.markdown(f"- **Asset Count:** {profile['asset_count']}")
            st.markdown(f"- **Has Insurance:** {'Yes' if profile['has_insurance'] else 'No'}")
            st.markdown(f"- **Has Chronic Illness:** {'Yes' if profile['has_chronic_illness'] else 'No'}")

    # Cluster Insights Tab
    with tabs[4]:
        cluster_data = recommendations["Cluster Data"]
        
        st.markdown("### Cluster Analysis")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_income = cluster_data['avg_mnth_income'].mean()
            st.metric("Average Cluster Income", f"KES {avg_income:,.2f}")
        with col2:
            insurance_rate = cluster_data['insurance'].mean() * 100
            st.metric("Insurance Adoption Rate", f"{insurance_rate:.1f}%")
        with col3:
            avg_age = cluster_data['age_of_respondent'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")

        # Feature importance visualization
        st.markdown("### Key Influential Features")
        feature_importance = recommendations["Feature Importance"]
        if feature_importance and feature_importance['sorted_features']:
            importance_df = pd.DataFrame(
                feature_importance['sorted_features'][:10], 
                columns=['Feature', 'Importance']
            )
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Most Influential Features'
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Feature Importance Score",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig)

        # Cluster distribution plots
        st.markdown("### Cluster Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                cluster_data, 
                x='avg_mnth_income',
                title='Income Distribution',
                nbins=30
            )
            st.plotly_chart(fig)
        with col2:
            fig = px.histogram(
                cluster_data,
                x='age_of_respondent', 
                title='Age Distribution',
                nbins=20,
                
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False
            )
            st.plotly_chart(fig)

#Action Button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_button = st.button('Generate Recommendations', 
                             type='primary', 
                             use_container_width=True)

if analyze_button:
    with st.spinner('Analyzing profile and generating recommendations...'):
        try:
            recommender = InsuranceRecommender(
                model_path="App/models/classifier.joblib",
                api_key=st.secrets["api_key"]
            )
            
            recommendations = recommender.get_customer_recommendations(
                preprocessed_data=preprocess_data, 
                customer_data=input_df
            )

            st.markdown("<h2 style='margin-top: 3rem;'>Analysis Results</h2>", 
                       unsafe_allow_html=True)
            
            # Create tabs with custom styling
            tab_style = """
                <style>
                .stTabs [data-baseweb="tab-list"] {
                    gap: 24px;
                }
                .stTabs [data-baseweb="tab"] {
                    padding: 10px 24px;
                    background-color: transparent !important;
                    color: white !important;
                    border: 1px solid rgba(128, 128, 128, 0.2);
                    border-radius: 4px;
                }
                .stTabs [data-baseweb="tab"]:hover {
                    background-color: rgba(255, 255, 255, 0.1) !important;
                }
                </style>
            """
            st.markdown(tab_style, unsafe_allow_html=True)
            
            create_dashboard(recommendations)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)



        
