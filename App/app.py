import streamlit as st
import pandas as pd

data = pd.read_csv('../Nakuru_FinAccess.csv')

st.write(data)
