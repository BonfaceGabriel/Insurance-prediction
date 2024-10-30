import streamlit as st
import pandas as pd

data = pd.read_csv('../Nakuru_FinAccess.csv')

st.write(data)
st.scatter_chart(data=data, x='avg_mnth_income', y='total_exp_per_month', x_label='Average Monthly Income', 
                y_label='Monthly Expenditure', color='insurance', size=None, width=None, height=None, use_container_width=True)