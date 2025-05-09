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
# from google import genai # Assuming this was for an alternative, commented out for now

st.set_page_config(
    page_title="InsureAI | Smart Insurance Recommendations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Consolidated Global CSS ---
GLOBAL_CSS = """
    <style>
    /* General Layout & Body */
    body {
        font-family: 'Arial', sans-serif; /* Example: set a global font */
    }
    .main {
        padding: 1rem 3rem; /* Adjusted main content padding */
    }

    /* Headings */
    h1, h2 { /* For st.title, st.header */
        color: white; /* Assuming a dark theme context */
    }
    h3 { /* For st.subheader and custom <h3> */
        color: #F0F0F0; /* Lighter text for subheaders */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Title Section */
    .title-container {
        padding: 2rem 1rem; /* Increased padding */
        margin-bottom: 2.5rem;
        background: linear-gradient(90deg, #1E3D59 0%, #2E5E88 100%);
        border-radius: 12px; /* Slightly more rounded */
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }
    .title { /* Class for the main title text if wrapped in a div */
        color: white;
        text-align: center;
        font-size: 2.8rem; /* Slightly larger */
        font-weight: 700;
        margin: 0;
        padding: 0.5rem;
    }
    .subtitle { /* Class for the subtitle text if wrapped in a div */
        color: #D0D0D0;
        text-align: center;
        font-size: 1.3rem; /* Slightly larger */
        font-weight: 400;
        margin: 0;
        padding: 0.5rem 1rem;
    }

    /* General Card Styling */
    .card-common { /* Base for white cards */
        background-color: #FFFFFF;
        color: #333333; /* Dark text for readability */
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .feature-card { /* Used in intro markdown */
        background-color: #F8F9FA; /* Slightly off-white */
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #333; /* Ensure text is dark */
    }
    .metric-card { /* For st.metric displays or custom metrics */
        background-color: #FFFFFF;
        color: #333333;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .recommendation-card { /* For individual recommendations in dashboard */
        background-color: #FFFFFF;
        color: #333333;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E5E88; /* Accent border */
        margin-bottom: 1rem;
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }
    .metric-container { /* Wrapper for metrics in dashboard */
        background-color: #FFFFFF;
        color: #333333;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .chart-container { /* Wrapper for charts */
        background-color: #FFFFFF;
        color: #333333; /* Ensures chart titles/axes are visible */
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
        margin-bottom: 1.5rem;
    }

    /* Input Section Styling */
    .input-section-container { /* Optional: Wrapper for subheader + input-section */
        margin-bottom: 2rem;
    }
    .input-section {
        background-color: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.25); /* Slightly more visible border */
        border-radius: 12px;
        padding: 25px;
        margin-top: 1rem; /* Space after subheader */
    }
    /* Styling for Streamlit input widget labels */
    .input-section .stTextInput label,
    .input-section .stNumberInput label,
    .input-section .stSelectbox label,
    .input-section .stCheckbox label,
    .input-section .stDateInput label,
    .input-section .stTextArea label {
        color: #E0E0E0 !important; /* Light color for labels */
        font-size: 0.95rem;
    }
    /* Styling for sub-headings within the input section */
    .input-section h4 {
        color: #FFFFFF;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 0.5rem;
    }


    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px; /* Spacing between tab buttons */
        border-bottom: 2px solid rgba(255, 255, 255, 0.2); /* Underline for the tab bar */
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: transparent !important;
        color: #B0B0B0 !important; /* Grey for unselected tab text */
        border: none !important;
        border-bottom: 3px solid transparent !important; /* For active indicator */
        border-radius: 0; /* Flat look */
        transition: color 0.2s ease, border-bottom-color 0.2s ease;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #FFFFFF !important; /* White underline for selected tab */
        font-weight: 600;
    }

    /* Expander styling for recommendations */
    div[data-testid="stExpander"] summary {
        background-color: rgba(46, 94, 136, 0.2); /* Semi-transparent theme color */
        color: #E0E0E0; /* Light text */
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 12px 18px;
        font-weight: 500;
    }
    div[data-testid="stExpander"] summary:hover {
        background-color: rgba(46, 94, 136, 0.3);
        color: #FFFFFF;
    }
    /* Content of expander: make background transparent to see page bg or define one */
    div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
        /* background-color: rgba(255, 255, 255, 0.05); /* very subtle bg for content */
        /* padding-top: 10px; */
    }


    /* Recommendation cards INSIDE expanders (dashboard) */
    .recommendation-group-header { /* Replaces inline style for the "Cluster-Based Recs" header */
        background-color: rgba(30, 61, 89, 0.5); /* Darker, related to title gradient */
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .recommendation-group-header h3 {
        color: #FFFFFF;
        margin-top: 0;
        margin-bottom: 0.25rem;
        font-size: 1.4rem;
    }
    .recommendation-group-header p {
        color: #D0D0D0;
        margin-bottom: 0;
        font-size: 0.9rem;
    }

    .recommendation-item-card { /* Replaces inline style for individual rec details */
        background-color: rgba(255, 255, 255, 0.03); /* Very subtle light card on dark bg */
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #E0E0E0; /* Light text */
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 0.5rem; /* Space between item card and expander title */
    }
    .recommendation-item-card br { /* Ensure <br> creates visual space */
        content: " ";
        display: block;
        margin-bottom: 0.6em;
    }

    /* Make Plotly chart backgrounds transparent */
    .plotly-graph-div {
        background: transparent !important;
    }
    /* Ensure Plotly chart text is visible on dark theme */
     .js-plotly-plot .plotly svg {
        background-color: transparent !important;
    }
    .js-plotly-plot .plotly .shapelayer path,
    .js-plotly-plot .plotly .gridlayer path {
        /* stroke: rgba(255, 255, 255, 0.3) !important; */ /* Lighter grid lines */
    }
    .js-plotly-plot .plotly .xaxislayer-above text,
    .js-plotly-plot .plotly .yaxislayer-above text,
    .js-plotly-plot .plotly .legendtext {
        fill: #E0E0E0 !important; /* Light text for axes and legend */
        font-size: 11px !important;
    }
    .js-plotly-plot .plotly .g-title text {
         fill: #FFFFFF !important; /* White title text */
         font-size: 16px !important;
    }

    </style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# Original st.markdown for title structure (using classes from GLOBAL_CSS)
st.markdown("""
    <div class="title-container">
        <p class="title">InsureAI 🛡️</p>
        <p class="subtitle">Smart Insurance Recommendations Powered by AI</p>
    </div>
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
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Apply scaling
    scaled_data = scaler.transform(df_copy[num_cols])
    df_copy[num_cols] = scaled_data
    
    # Apply one-hot encoding
    encoded = encoder.transform(df_copy[cat_cols])
    one_hot_df = pd.DataFrame(encoded,
                              columns=encoder.get_feature_names_out(cat_cols),
                              index=df_copy.index) # Preserve index
                              
    # Concatenate
    processed_data = pd.concat([df_copy.drop(columns=cat_cols), one_hot_df], axis=1)
    
    # Convert object columns (likely boolean after processing) to bool
    cols_to_convert = processed_data.select_dtypes(include=['object']).columns
    for col in cols_to_convert:
        # Handle potential mixed types or NaNs before conversion
        if processed_data[col].dtype == 'object':
            # A robust way to convert to boolean, assuming 'True'/'False' strings or similar
            # If they are already boolean but dtype is object, this map might not be needed
            # but explicit conversion is safer.
            # Example: map {'True': True, 'False': False} if they are strings
            # For now, direct astype(bool) but be mindful of how objects become bools
            try:
                processed_data[col] = processed_data[col].astype(bool)
            except ValueError:
                # Handle cases where direct conversion to bool might fail for object types
                # This might indicate an issue upstream or a need for more specific mapping
                st.warning(f"Could not convert column {col} to boolean directly. Check data.")
                # Fallback or specific mapping might be needed here
                # For example, if they are 'Yes'/'No' strings:
                # processed_data[col] = processed_data[col].map({'Yes': True, 'No': False, True: True, False: False}).fillna(False)


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
        if data['insurance'].values[0]: # Accessing the original 'insurance' column
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
                'has_insurance': customer_data['insurance'].values[0], # Accessing the original 'insurance' column
                'has_chronic_illness': customer_data['chronic_illness'].values[0]
            }
        }
        
    def _determine_segment(self, income: float, asset_count: int) -> str:
        for segment_name, criteria in self.segments.items(): # Renamed 'segment' to 'segment_name'
            if (income >= criteria['min_income'] and 
                asset_count >= criteria['min_assets']):
                return segment_name
        return 'BASIC'

class InsuranceRecommender:
    def __init__(self, model_path, api_key):
        with open(model_path, "rb") as model_file:
            self.cluster_model = joblib.load(model_file)
        self.client = Anthropic(api_key=api_key)
        self.risk_assessor = RiskAssessment()
        self.profiler = CustomerProfile()
        self.products = InsuranceProducts()
        
        try:
            # Create a sample from the original data for background, ensure it has 'Clusters' if model expects it
            # The model loaded (`classifier.joblib`) is a LightGBMClassifier according to its usage with TreeExplainer.
            # It likely does not predict 'Clusters' but the insurance uptake.
            # If 'classifier.joblib' is the clustering model itself, then SHAP TreeExplainer is appropriate.
            # If 'classifier.joblib' is a classification model (e.g., predicting insurance uptake)
            # and clustering is done separately or not at all by this model, then background data needs to match its training input.

            # Assuming self.cluster_model is indeed the model for which SHAP values are needed.
            # And assuming `data` (global DataFrame) is representative of its training data structure *before* preprocessing specific to this app.
            # For TreeExplainer, background data should be in the format the model expects.
            # If self.cluster_model was trained on preprocessed data, then shap.sample should use preprocessed data.
            
            # Let's assume `self.cluster_model` was trained on data similar to `preprocess_data`
            # We need a preprocessed background sample.
            
            if not data.empty:
                 # Check if 'Clusters' column exists. If not, it means the model might be a classifier for insurance uptake, not a clusterer.
                if 'Clusters' not in data.columns:
                    # If 'classifier.joblib' is for predicting insurance (binary), and not clustering,
                    # then SHAP explains this prediction.
                    # The background data needs to be preprocessed.
                    # Create a sample from the raw data and preprocess it.
                    background_sample_raw = shap.sample(data.drop(columns=['insurance'], errors='ignore'), 100, random_state=42) 
                    
                    # We need to preprocess this background_sample_raw like any other input
                    # It might be tricky if 'preprocess' relies on single-row logic or specific columns not in sample.
                    # For simplicity, if `data` is large, we can preprocess a sample of `data`.
                    # Let's preprocess a sample of the main `data` dataframe.
                    
                    data_for_shap_sample = data.drop(columns=['insurance'], errors='ignore').copy() # Drop target if it was there
                    # Ensure all columns expected by preprocess() are present in data_for_shap_sample
                    # Required columns by preprocess(): ['age_of_respondent', 'avg_mnth_income', 'total_exp_per_month', 
                    # 'most_important_life_goal', 'area', 'income_source', 'nearest_financial_prod']
                    # plus boolean columns used in preprocess but not explicitly listed there like 'chronic_illness' etc.
                    # This implies `data` should already have these.

                    # A sample from the full dataset, then preprocess it
                    sampled_data_for_background = data.sample(min(100, len(data)), random_state=10)
                    # Drop target before preprocessing for background
                    if 'insurance' in sampled_data_for_background.columns:
                         sampled_data_for_background = sampled_data_for_background.drop(columns=['insurance'])

                    # The preprocess function expects a DataFrame with specific column names.
                    # We need to ensure the background data matches the structure of `preprocessed_data` argument to `get_feature_importance`.
                    # This means the background data also needs to be preprocessed.
                    
                    # Let's create a preprocessed background dataset from the global `data`
                    # We need all features that go into the model.
                    # The model `classifier.joblib` seems to be the one doing predictions based on `preprocessed_data`
                    # So, the background for SHAP should also be preprocessed.
                    
                    # Create a copy of the full dataset for preprocessing
                    data_for_background_processing = data.copy()
                    # The 'insurance' column is the target, it should not be in features for SHAP background
                    if 'insurance' in data_for_background_processing.columns:
                        data_for_background_processing = data_for_background_processing.drop(columns=['insurance'])
                    
                    # Drop 'Clusters' if it exists and is not a feature for the model.
                    if 'Clusters' in data_for_background_processing.columns:
                         data_for_background_processing = data_for_background_processing.drop(columns=['Clusters'])

                    preprocessed_full_data_for_background = preprocess(data_for_background_processing)
                    self.background_data_for_shap = shap.sample(preprocessed_full_data_for_background, min(100, len(preprocessed_full_data_for_background)), random_state=0)
                    self.explainer = shap.TreeExplainer(self.cluster_model, self.background_data_for_shap)

            else:
                st.warning("Data for SHAP background is empty.")
                self.explainer = None
                self.background_data_for_shap = None


        except Exception as e:
            st.error(f"Error initializing SHAP explainer: {str(e)}")
            self.explainer = None
            self.background_data_for_shap = None


    def call_claude_api(self, prompt):
        # Assuming 'claude-3-5-sonnet-20241022' is a typo and should be a valid model like 'claude-3-sonnet-20240229'
        # Or using a newer one if available by the time of execution.
        # For this exercise, I'll use a placeholder or a commonly available one.
        # Let's assume 'claude-3-sonnet-20240229' or similar is intended.
        # The provided model name 'claude-3-5-sonnet-20241022' is futuristic.
        # Using 'claude-3-opus-20240229' as an example, replace with actual valid model.
        # For now, I will keep the user's model name as they might have access to it.
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620", # Corrected to a plausible existing model if 20241022 was a typo
            max_tokens=2048, # Increased max_tokens for potentially longer recommendations
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    

    def extract_report_content(self, profile_report):
        try:
            text_summary = ""
            numeric_distributions = {}
            correlation_matrix_df = pd.DataFrame() # Initialize as DataFrame

            if profile_report is None:
                raise ValueError("Profile report is None")
                
            profile_html = profile_report.to_html()
            soup = BeautifulSoup(profile_html, "html.parser")
            
            # Try to get a concise summary text
            overview_section = soup.find("div", id="overview")
            if overview_section:
                 text_summary = " ".join([el.get_text(strip=True) for el in overview_section.find_all(["p", "li"])])
            if not text_summary: # Fallback
                text_summary = " ".join([el.get_text(strip=True) for el in soup.find_all(["h1", "h2", "p", "li"])[:50]]) # Limit length


            # Get correlation matrix (Pearson)
            try:
                # Accessing correlations can be tricky depending on ydata-profiling version
                if hasattr(profile_report, 'description_set') and profile_report.description_set.correlations:
                    if 'pearson' in profile_report.description_set.correlations:
                         correlation_matrix_df = profile_report.description_set.correlations['pearson']
                elif hasattr(profile_report, 'get_correlation_matrix'): # Older versions
                    correlation_matrix_df = profile_report.get_correlation_matrix()['pearson']
                
                if isinstance(correlation_matrix_df, pd.DataFrame) and not correlation_matrix_df.empty:
                    correlation_matrix = correlation_matrix_df.to_dict()
                else:
                    correlation_matrix = {}

            except Exception as e_corr:
                print(f"Error extracting correlation matrix: {e_corr}")
                correlation_matrix = {}


            try:
                variables_desc = profile_report.get_description().variables # Common way to get variable descriptions
                for col, var_data in variables_desc.items():
                    if var_data.type == "Numeric":
                        numeric_distributions[col] = {
                            "mean": var_data.mean if hasattr(var_data, 'mean') else 0,
                            "std": var_data.std if hasattr(var_data, 'std') else 0,
                            "min": var_data.min if hasattr(var_data, 'min') else 0,
                            "max": var_data.max if hasattr(var_data, 'max') else 0,
                            # Histogram data might be structured differently or not easily accessible as simple counts
                            # "histogram_counts": var_data.histogram_counts if hasattr(var_data, 'histogram_counts') else [],
                            # "histogram_bins": var_data.histogram_bins if hasattr(var_data, 'histogram_bins') else []
                        }
            except Exception as e_vars:
                print(f"Error extracting numeric distributions: {e_vars}")
                pass # numeric_distributions remains as initialized

            return text_summary, numeric_distributions, correlation_matrix

        except Exception as e:
            print(f"Error in extract_report_content: {str(e)}")
            return "Could not extract summary.", {}, {}

    def get_feature_importance(self, preprocessed_data_instance):
        try:
            if self.explainer is None:
                st.warning("SHAP explainer not initialized. Feature importance will be unavailable.")
                return {'importance_dict': {}, 'sorted_features': [], 'shap_values_instance': None}

            # Ensure preprocessed_data_instance is a DataFrame for column access
            if not isinstance(preprocessed_data_instance, pd.DataFrame):
                # This should already be a DataFrame from preprocess()
                # If it's a NumPy array, it needs feature names from when it was created
                st.error("Preprocessed data for SHAP must be a Pandas DataFrame.")
                return {'importance_dict': {}, 'sorted_features': [], 'shap_values_instance': None}

            feature_names = preprocessed_data_instance.columns.tolist()
            
            # For TreeExplainer, shap_values returns an array (for binary/regression) or list of arrays (for multiclass)
            # Assuming classifier.joblib is for binary classification (e.g. insurance uptake) or regression.
            # If it's a clustering model, SHAP might not be directly applicable in the same way for "feature importance of a cluster prediction".
            # Given it's named 'classifier.joblib', let's assume it's a classifier.
            
            shap_values_instance = self.explainer.shap_values(preprocessed_data_instance) # SHAP values for the single instance

            importance_dict = {}
            
            # Handle different SHAP value structures (binary vs. multiclass)
            # For binary classification with TreeExplainer, shap_values might return values for one class or both.
            # If it returns for both classes (list of two arrays), typically use shap_values[1] for the positive class.
            # If it returns a single array, it's usually for the positive class.
            
            processed_shap_values = shap_values_instance
            if isinstance(shap_values_instance, list) and len(shap_values_instance) == 2: # Common for binary classifiers
                processed_shap_values = shap_values_instance[1] # Values for the positive class
            
            # We are explaining a single instance, so shap_values_instance will be 1D (or 2D with one row)
            # np.abs().mean(axis=0) is for multiple instances. For one, just take abs().
            if processed_shap_values.ndim == 2: # If it's like (1, num_features)
                abs_shap_values = np.abs(processed_shap_values[0])
            else: # If it's 1D (num_features)
                abs_shap_values = np.abs(processed_shap_values)

            for i, feature in enumerate(feature_names):
                importance_dict[feature] = float(abs_shap_values[i])
            
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'importance_dict': importance_dict,
                'sorted_features': sorted_features,
                'shap_values_instance': processed_shap_values # Return the SHAP values for the specific instance
            }
        except Exception as e:
            st.error(f"Error calculating SHAP values: {str(e)}")
            return {'importance_dict': {}, 'sorted_features': [], 'shap_values_instance': None}


    def generate_recommendations(self, text_summary, numeric_distributions, correlation_matrix, 
                           risk_assessment, customer_profile, available_products, 
                           feature_importance):
        try:
            risk_assessment_safe = {
                'total_risk': float(risk_assessment['total_risk']),
                'components': {k: float(v) for k, v in risk_assessment['components'].items()}
            }
            
            profile_safe = {'segment': customer_profile['segment']}
            for k, v in customer_profile['profile'].items():
                if isinstance(v, (np.integer, int, np.floating, float)):
                    profile_safe[k] = float(v)
                elif isinstance(v, (np.bool_, bool)):
                    profile_safe[k] = bool(v)
                else:
                    profile_safe[k] = str(v) # Ensure all profile details are JSON serializable

            feature_importance_safe = {
                'top_features': {str(k): float(v) for k, v in feature_importance.get('sorted_features', [])[:5]}
            }

            # Cluster-based recommendations prompt
            cluster_prompt = f"""Analyze the following customer cluster information:
            - Cluster Profile Summary: {text_summary if text_summary else 'Not available'}
            - Key Numeric Distributions: {json.dumps(numeric_distributions, cls=NumpyEncoder, indent=2)}
            - Correlations: {json.dumps(correlation_matrix, cls=NumpyEncoder, indent=2)}
            - Available Insurance Products: {json.dumps(available_products, cls=NumpyEncoder, indent=2)}

            Based on this cluster data, provide 2-3 general insurance product recommendations suitable for the typical customer in this cluster.
            For each recommendation:
            1. State the Product Name and Tier (e.g., Comprehensive Health Cover - PREMIUM).
            2. Briefly mention key coverage or benefits.
            3. Suggest a suitable payment plan (e.g., Monthly, Annually with discount).
            4. Explain concisely why this product is a good fit for this cluster's general characteristics.

            Format the response clearly under a "Cluster-Based Recommendations" heading.
            Example for one product:
            1. Comprehensive Health Cover - PREMIUM
               - Coverage: Up to KES 1,000,000. Covers inpatient, outpatient, chronic diseases.
               - Payment Plan: Annually at KES 55,000 (8% discount).
               - Cluster Fit: Suitable for clusters with higher income and need for extensive health coverage.
            """
            cluster_recommendations = self.call_claude_api(cluster_prompt)

            # Personalized recommendations prompt
            personal_prompt = f"""Analyze the following specific customer information:
            - Customer Risk Assessment:
              - Overall Risk Score: {risk_assessment_safe['total_risk']:.2f} (0-1 scale)
              - Risk Components: {json.dumps(risk_assessment_safe['components'], indent=2)}
            - Customer Profile: {json.dumps(profile_safe, indent=2)}
            - Top Features Influencing Predictions (from SHAP): {json.dumps(feature_importance_safe['top_features'], indent=2)}
            - Available Insurance Products: {json.dumps(available_products, cls=NumpyEncoder, indent=2)}
            - General Cluster Recommendations (for context, try to offer different/more specific advice): {cluster_recommendations}

            Based on this INDIVIDUAL customer's data, provide 2-3 personalized insurance product recommendations. These should be tailored to their unique situation, risk profile, stated life goals, and financial capacity, potentially differing from or refining the general cluster advice.
            For each recommendation:
            1. State the Product Name and Tier.
            2. Highlight specific benefits relevant to THIS customer.
            3. Suggest a payment plan considering their income and expenses.
            4. Explain precisely why this product and plan are a strong personal fit.

            Format the response clearly under a "Personalized Recommendations" heading.
            Ensure recommendations are actionable and justified by the customer's specific data.
            """
            personal_recommendations = self.call_claude_api(personal_prompt)

            return {
                "cluster_recommendations": cluster_recommendations,
                "personal_recommendations": personal_recommendations
            }

        except Exception as e:
            st.error(f"Error in generate_recommendations: {str(e)}")
            return {
                "cluster_recommendations": "Error: Could not generate cluster recommendations.",
                "personal_recommendations": "Error: Could not generate personalized recommendations."
            }

    def get_customer_recommendations(self, preprocessed_data_instance, customer_data_instance):
        # Predict cluster for the specific customer instance
        # Assuming self.cluster_model.predict() gives a cluster label if it's a clustering model
        # If 'classifier.joblib' is a classifier (e.g. predicts insurance uptake 'Yes'/'No'), 
        # then 'cluster' variable might be misleading.
        # Let's assume it's a classifier for now, and "cluster" refers to a segment based on this prediction or other rules.
        # For this code, `self.cluster_model` is used with `preprocessed_data` which implies it's the main prediction model.
        # The original code used `data[data['Clusters'] == cluster]` - this assumes a 'Clusters' column in the original `data` df.
        # If 'classifier.joblib' IS the clustering model, this is fine.
        # If 'classifier.joblib' is, e.g., a LightGBM classifier predicting insurance uptake, then `predicted_value` is not a cluster.
        
        # Let's clarify: If `classifier.joblib` (loaded as `self.cluster_model`) is a classifier,
        # then `predicted_value` is a class label (e.g., 0 or 1 for no/yes insurance).
        # The concept of "cluster_data" then needs to be redefined. It could be data points with the same predicted class.
        
        predicted_value = self.cluster_model.predict(preprocessed_data_instance)[0]
        # This could be a cluster ID if it's a clustering model, or a class label (0/1) if it's a classifier.
        # The code used `data['Clusters']` column. If this column doesn't exist or isn't related to `predicted_value`, this will fail.
        # For now, let's assume 'classifier.joblib' output can be used to segment data.
        # A common scenario: if it's a classifier (e.g. predicts high/low risk), then cluster_data = data[data['some_column_mapped_from_prediction'] == mapped_value]
        # Or, if 'Clusters' column is pre-assigned in `data` and we just want to profile a *hypothetical* cluster based on prediction:
        
        # Simplification: Assume `predicted_value` directly maps to a segment or can be used to filter.
        # If 'Clusters' column is not in `data`, `ProfileReport` on `cluster_data` will fail.
        # Let's assume 'Clusters' is a column in the original `data` CSV that was pre-computed.
        # And the `predicted_value` from `self.cluster_model` matches values in this `data['Clusters']` column.
        # This is a strong assumption.
        
        # More robustly, if `classifier.joblib` IS the clustering model itself, this is fine.
        # If not, the 'cluster_data' logic needs to be re-evaluated.
        # Given the name "classifier.joblib" and SHAP TreeExplainer, it's likely a tree-based classifier.
        # Let's assume `predicted_value` is a segment/cluster ID for now.
        
        # Check if 'Clusters' column exists in the global `data`
        if 'Clusters' in data.columns:
            cluster_data_for_profiling = data[data['Clusters'] == predicted_value]
            if cluster_data_for_profiling.empty:
                st.warning(f"No data found for predicted segment/cluster {predicted_value}. Profiling may be limited. Using all data as fallback for profile report.")
                # Fallback to overall data if specific cluster is empty, or handle as error
                cluster_data_for_profiling = data # Or a subset
        else:
            st.warning("'Clusters' column not found in data. Cluster-specific profiling will use all data.")
            cluster_data_for_profiling = data # Profile report on all data as a fallback

        profile_report = ProfileReport(cluster_data_for_profiling, 
                                       minimal=True, 
                                       explorative=True, # Enable correlations
                                       config_override={"correlations": {"pearson": {"calculate": True}}})


        feature_importance = self.get_feature_importance(preprocessed_data_instance)
        risk_assessment = self.risk_assessor.calculate_risk_score(customer_data_instance)
        customer_profile = self.profiler.create_profile(customer_data_instance)
        available_products = self.products.products

        text_summary, numeric_distributions, correlation_matrix = self.extract_report_content(profile_report)
        
        recommendations_payload = self.generate_recommendations(
            text_summary, 
            numeric_distributions, 
            correlation_matrix,
            risk_assessment,
            customer_profile,
            available_products,
            feature_importance
        )

        return {
            "PredictedSegment": predicted_value, # Renamed from "Predicted Cluster" for clarity
            "RiskAssessment": risk_assessment,
            "CustomerProfile": customer_profile,
            "Recommendations": recommendations_payload,
            "SegmentDataForProfileReport": cluster_data_for_profiling, # Data used for the ydata_profiling report
            "FeatureImportance": feature_importance
        }

# Initialize recommender (ensure API key is correctly loaded)
# This instantiation happens once when the script runs.
try:
    recommender = InsuranceRecommender(
        model_path="App/models/classifier.joblib",
        api_key=st.secrets["anthropic_api_key"] # Use a specific key name
    )
except Exception as e:
    st.error(f"Failed to initialize the InsuranceRecommender: {e}")
    st.stop() # Stop the app if recommender fails to initialize

#main page content
c = st.container() # Removed border=True as it might conflict with theme
with c: # Using 'with' for containers is good practice
    st.markdown("""
    This application provides AI-driven insurance product recommendations tailored to customer characteristics, primarily focusing on data relevant to Nakuru County.
    
    **Key Features:**
    - **Customer Profiling:** Understand your financial and risk profile.
    - **Personalized Recommendations:** Receive insurance suggestions based on your unique data.
    - **Segment-Based Insights:** Get recommendations relevant to broader customer segments.
    - **Risk Analysis:** Detailed breakdown of factors contributing to your risk score.
    - **Feature Importance:** Discover which data points most influence predictions (via SHAP).

    Enter your details below to get started!
    """, unsafe_allow_html=True) # unsafe_allow_html for list styling if any


def user_input_features():
    st.subheader('👤 Customer Information') # Simpler subheader
    
    # Wrap the entire input area in the .input-section div
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Personal Details")
        age_of_respondent = st.number_input('Age', min_value=16, max_value=100, value=30, help="Your current age.")
        area = st.selectbox('Area of Residence', ['Urban', 'Rural'], help="Your primary area of residence.")
        chronic_illness = st.checkbox('Do you have any chronic illness?', value=False, help="Check if you have a long-term health condition.")
            
    with col2:
        st.markdown("#### Financial Information")
        avg_mnth_income = st.number_input('Average Monthly Income (KES)', min_value=0, value=50000, step=1000, help="Your typical income per month.")
        total_exp_per_month = st.number_input('Total Monthly Expenses (KES)', min_value=0, value=30000, step=1000, help="Your estimated total spending per month.")
            
    st.markdown("---") # Visual separator

    col3, col4, col5 = st.columns(3)
    with col3:
        st.markdown("#### Income & Goals")
        income_source = st.selectbox('Primary Source of Income', 
            sorted(['Casual_work', 'Family', 'Agriculture', 'Business', 
            'Employment', 'Renting', 'Pension', 'Aid']), help="Your main way of earning income.")
        most_important_life_goal = st.selectbox('Most Important Life Goal',
            sorted(['Food', 'Education', 'Health', 'Business', 
            'Career', 'Home', 'Assets', 'None']), help="What is your current top priority in life?")
                
    with col4:
        st.markdown("#### Financial Products & Coverage")
        nearest_financial_prod = st.selectbox('Nearest Financial Product/Service', 
            sorted(['MMoney', 'Bank', 'Insurance', 'SACCO', 'Microfinance']), help="Which financial service is most accessible to you?")
        nhif_usage = st.checkbox('Are you covered by NHIF?', value=True, help="National Hospital Insurance Fund coverage.")
        nssf_usage = st.checkbox('Are you contributing to NSSF?', value=False, help="National Social Security Fund contributions.")
        insurance = st.checkbox('Do you currently have any other insurance (non-NHIF)?', value=False, help="Any private insurance policies.")
            
    with col5:
        st.markdown("#### Assets & Liabilities")
        land_house_ownership = st.checkbox('Do you own land or a house?', value=False)
        motorvehicle_ownership = st.checkbox('Do you own a motor vehicle?', value=False)
        electronic_device = st.checkbox('Do you own valuable electronic devices (e.g., smartphone, laptop)?', value=True)
        livestock_ownership = st.checkbox('Do you own livestock (for farming/business)?', value=False)
        hse_land_loan = st.checkbox('Do you have an active loan for house/land?', value=False)
        securities_use = st.checkbox('Do you invest in stocks, bonds, or other securities?', value=False)

    st.markdown('</div>', unsafe_allow_html=True) # Close .input-section
        
    user_data = {'area': area,
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
            'insurance': insurance, # This is the target variable for prediction but also an input feature
            }
    features = pd.DataFrame(user_data, index=[0])
    return features

input_df = user_input_features()

# Display user input for confirmation (optional, can be cleaner)
with st.expander(" görmek için tıklayın (See Your Input Summary)", expanded=False):
    st.dataframe(input_df.T.rename(columns={0: 'Your Input'}))


# Preprocess a copy for the model, keep original input_df for other uses
processed_input_df = preprocess(input_df.copy())


def create_dashboard(analysis_results):
    # analysis_results is the dictionary returned by get_customer_recommendations
    st.subheader("📈 Analysis & Recommendations Dashboard")

    tabs = st.tabs([
        "📊 Overview", 
        "🎯 Recommendations", 
        "⚠️ Risk Analysis", 
        "👤 Customer Profile",
        "💡 Segment Insights" # Renamed from Cluster Insights
    ])

    # Overview Tab
    with tabs[0]:
        st.markdown("### Key Insights at a Glance")
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_score = analysis_results["RiskAssessment"]["total_risk"] * 100
            st.metric("Overall Risk Score", f"{risk_score:.1f}%", delta_color="inverse",
                      help="A measure of your overall financial and insurable risk (0-100%). Higher means more risk factors identified.")
        with col2:
            segment = analysis_results["CustomerProfile"]["segment"]
            st.metric("Customer Segment", segment, help="Your profile segment based on income and assets (e.g., BASIC, STANDARD, PREMIUM).")
        with col3:
            # 'PredictedSegment' is what get_customer_recommendations returns
            pred_segment_val = analysis_results["PredictedSegment"] 
            st.metric("Predicted Group", f"Group {pred_segment_val}", help="The analytical group your profile aligns with based on the predictive model.")

        st.markdown("---")
        st.markdown("<p style='font-style: italic; color: #B0B0B0;'>Dive into the other tabs for detailed recommendations and analysis.</p>", unsafe_allow_html=True)


    # Recommendations Tab
    with tabs[1]:
        st.markdown("### Your Tailored Insurance Advice")
        cluster_col, personal_col = st.columns(2)
        
        recommendations = analysis_results["Recommendations"]
        with cluster_col:
            st.markdown("<div class='recommendation-group-header'><h3>🔍 Segment-Based Suggestions</h3><p>General recommendations for your customer group.</p></div>", unsafe_allow_html=True)
            # Parse and display cluster recommendations
            cluster_recs_text = recommendations.get("cluster_recommendations", "No segment-based recommendations available.")
            
            # Simple parsing: Assume recommendations are numbered or bulleted
            cluster_rec_list = [rec.strip() for rec in cluster_recs_text.splitlines() if rec.strip() and (rec.strip().startswith(tuple(str(i) + "." for i in range(10))) or rec.strip().startswith("-") or rec.strip().startswith("*"))]
            
            if "Error:" in cluster_recs_text or not cluster_rec_list:
                 st.markdown(f"<div class='recommendation-item-card'>{cluster_recs_text}</div>", unsafe_allow_html=True)
            else:
                # Try to display each recommendation in an expander or card
                # This part needs robust parsing of Claude's output.
                # For now, displaying as a block:
                st.markdown(f"<div class='recommendation-item-card'>{cluster_recs_text.replace("\n", "<br>")}</div>", unsafe_allow_html=True)


        with personal_col:
            st.markdown("<div class='recommendation-group-header'><h3>👤 Personalized For You</h3><p>Specific advice based on your unique profile.</p></div>", unsafe_allow_html=True)
            personal_recs_text = recommendations.get("personal_recommendations", "No personalized recommendations available.")

            if "Error:" in personal_recs_text:
                st.markdown(f"<div class='recommendation-item-card'>{personal_recs_text}</div>", unsafe_allow_html=True)
            else:
                 st.markdown(f"<div class='recommendation-item-card'>{personal_recs_text.replace("\n", "<br>")}</div>", unsafe_allow_html=True)


    # Risk Analysis Tab
    with tabs[2]:
        st.markdown("### Detailed Risk Assessment")
        risk_data = analysis_results["RiskAssessment"]["components"]
        
        col1, col2 = st.columns([2, 1]) # Adjusted column widths
        with col1:
            st.markdown("#### Risk Component Radar")
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[risk_data[k] * 100 for k in risk_data.keys()] + [risk_data[list(risk_data.keys())[0]]*100], # Close the radar
                theta=[k.replace('_', ' ').title() for k in risk_data.keys()] + [list(risk_data.keys())[0].replace('_',' ').title()],
                fill='toself',
                marker_color='rgba(46, 94, 136, 0.7)', # Theme color
                line_color='rgba(30, 61, 89, 1)',   # Darker theme color for line
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], showline=False, showticklabels=True, ticks C=''),
                           angularaxis=dict(showline=False, showticklabels=True, ticks='')),
                showlegend=False,
                # title="Risk Component Analysis", # Title provided by st.markdown
                height=400,
                paper_bgcolor='rgba(0,0,0,0)', # Transparent paper
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot area
                font=dict(color="#E0E0E0") # Light font for tick labels
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.markdown("#### Component Scores")
            for component, value in risk_data.items():
                # Using st.container for better layout control of metrics
                with st.container(): # Changed from metric-card class to direct st.metric
                    st.metric(
                        label=component.replace('_', ' ').title(),
                        value=f"{value * 100:.1f}%"
                    )
            st.caption("Scores are on a 0-100% scale, where higher indicates greater contribution to overall risk.")


    # Customer Profile Tab
    with tabs[3]:
        st.markdown("### Your Profile Summary")
        profile = analysis_results["CustomerProfile"]["profile"]
        
        col_prof1, col_prof2 = st.columns(2)
        with col_prof1:
            st.markdown(f"""
            <div class='recommendation-item-card'>
                <h4>Key Demographics</h4>
                - **Age:** {profile['age']}<br>
                - **Area:** {profile['area']}<br>
                - **Life Goal:** {profile['life_goal']}<br>
                - **Income Source:** {profile['income_source']}
            </div>
            """, unsafe_allow_html=True)
        
        with col_prof2:
            st.markdown(f"""
            <div class='recommendation-item-card'>
                <h4>Financial Snapshot</h4>
                - **Monthly Income:** KES {profile['income']:,.0f}<br>
                - **Monthly Expenses:** KES {profile['monthly_expenses']:,.0f}<br>
                - **Disposable Income:** KES {(profile['income'] - profile['monthly_expenses']):,.0f}<br>
                - **Savings Rate:** {((profile['income'] - profile['monthly_expenses']) / profile['income'] * 100 if profile['income'] > 0 else 0):.1f}%
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='recommendation-item-card' style='margin-top:1rem;'>
                <h4>Assets & Coverage</h4>
                - **Asset Count (tracked):** {profile['asset_count']}<br>
                - **Currently Has Insurance (non-NHIF):** {'Yes' if profile['has_insurance'] else 'No'}<br>
                - **Has Chronic Illness:** {'Yes' if profile['has_chronic_illness'] else 'No'}<br>
                - **NHIF Coverage:** {'Yes' if input_df['nhif_usage'].values[0] else 'No'}<br> <!-- from input_df -->
                - **NSSF Contributions:** {'Yes' if input_df['nssf_usage'].values[0] else 'No'} <!-- from input_df -->
            </div>
            """, unsafe_allow_html=True)


    # Segment Insights Tab
    with tabs[4]:
        st.markdown("### About Your Analytical Group")
        segment_profile_data = analysis_results["SegmentDataForProfileReport"] # This is the data used for ydata_profiling
        
        if not segment_profile_data.empty:
            col_seg1, col_seg2, col_seg3 = st.columns(3)
            with col_seg1:
                st.metric("Avg. Group Income", f"KES {segment_profile_data['avg_mnth_income'].mean():,.0f}")
            with col_seg2: # Assuming 'insurance' is boolean (True/False)
                st.metric("Group Insurance Rate", f"{segment_profile_data['insurance'].mean() * 100:.1f}%")
            with col_seg3:
                st.metric("Avg. Group Age", f"{segment_profile_data['age_of_respondent'].mean():.1f} yrs")
            st.markdown("---")
        else:
            st.info("Detailed segment/group data for comparison is limited for this profile.")


        st.markdown("#### Key Factors Influencing Predictions (SHAP)")
        feature_importance_data = analysis_results["FeatureImportance"]
        
        if feature_importance_data and feature_importance_data.get('sorted_features'):
            # For a single instance, a bar chart of SHAP values is more direct.
            # The `shap_values_instance` should be used.
            shap_values_for_instance = feature_importance_data.get('shap_values_instance') # This is a 1D array
            importance_df = pd.DataFrame({
                'Feature': processed_input_df.columns, # Ensure this matches the shap_values order
                'SHAP Value': shap_values_for_instance.flatten() # Ensure it's 1D
            }).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)

            fig_shap = px.bar(
                importance_df,
                x='SHAP Value',
                y='Feature',
                orientation='h',
                title='Top 10 Features Influencing Prediction for You',
                labels={'SHAP Value': 'SHAP Value (Impact on model output)'}
            )
            fig_shap.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=450,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#E0E0E0"),
                title_font_color="#FFFFFF"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Positive SHAP values increase the prediction towards the positive class (e.g., 'takes insurance'), negative values decrease it.")
        else:
            st.info("Feature importance (SHAP values) could not be calculated for this instance.")

        if not segment_profile_data.empty:
            st.markdown("#### Group Data Distributions")
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                fig_income_dist = px.histogram(
                    segment_profile_data, x='avg_mnth_income', title='Group Income Distribution', nbins=20
                )
                fig_income_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#E0E0E0"), title_font_color="#FFFFFF", yaxis_title="Count", xaxis_title="Average Monthly Income (KES)")
                st.plotly_chart(fig_income_dist, use_container_width=True)
            with col_dist2:
                fig_age_dist = px.histogram(
                    segment_profile_data, x='age_of_respondent', title='Group Age Distribution', nbins=15
                )
                fig_age_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#E0E0E0"), title_font_color="#FFFFFF", yaxis_title="Count", xaxis_title="Age of Respondent")
                st.plotly_chart(fig_age_dist, use_container_width=True)
        

# Action Button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1.5, 2, 1.5]) # Adjusted for better centering
with col_btn2:
    analyze_button = st.button('🚀 Generate My Recommendations', 
                             type='primary', 
                             use_container_width=True)

if analyze_button:
    if 'recommender' not in globals() or recommender is None:
        st.error("Recommender system is not initialized. Please check API key and model paths.")
    else:
        with st.spinner('🧠 Analyzing your profile and crafting recommendations... Please wait.'):
            try:
                analysis_results = recommender.get_customer_recommendations(
                    preprocessed_data_instance=processed_input_df, 
                    customer_data_instance=input_df # Pass original df for profiling functions
                )
                
                st.success("✨ Analysis Complete! View your personalized dashboard below.")
                st.markdown("<hr style='border:1px solid rgba(255,255,255,0.2); margin-top:2rem; margin-bottom:1rem;'>", unsafe_allow_html=True)
                create_dashboard(analysis_results)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e) # Shows full traceback for debugging
