import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


data = pd.read_csv('App/data/Nakuru_FinAccess1.csv')
pr = ProfileReport(data, config_file='App/configs_and_styles/config.yml')

st_profile_report(pr)