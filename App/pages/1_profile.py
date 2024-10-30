import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


data = pd.read_csv('../Nakuru_FinAccess1.csv')
pr = ProfileReport(data, config_file='/home/gabriel/financial-dataset/Insurance-prediction/App/pages/config.yml')

st_profile_report(pr)