import streamlit as st
import pandas as pd
import plost
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards

data = pd.read_csv('../Nakuru_FinAccess1.csv')

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('/home/gabriel/financial-dataset/Insurance-prediction/App/pages/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Dashboard `version 1`')

st.sidebar.subheader('Donut Chart parameters')
attribute = st.sidebar.selectbox('Select data', ('most_important_life_goal', 'income_source', 'nearest_financial_prod'))

st.sidebar.subheader('Scatter chart size')
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by [Bonface Odhiambo]
''')

count = data['area'].value_counts().to_list()
urban_pop = count[0]
rural_pop = count[1]
total = data['area'].count()

#Row A
col1, col2, col3 = st.columns(3)
col1.metric('Urban House Holds', f'{urban_pop} houses')
col2.metric('Rural House Holds', f'{rural_pop} houses') 
col3.metric('Total House Holds', f'{total} houses')
style_metric_cards(
    background_color = "#000000",
    border_size_px = 2,
    border_color = "#CCC",
    border_radius_px = 5,
    border_left_color = "#9AD8E1",
    box_shadow = True)

#Row B
c1, c2 = st.columns((7, 3))
with c1:
    st.subheader('Age Distribution')
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.hist(data['age_of_respondent'])
    st.pyplot(fig)

with c2:
    st.subheader('Donut Chart')
    value_counts = data[attribute].value_counts()
    percentages = value_counts / value_counts.sum() * 100
    percentages = round(percentages, 2)
    df_plot = pd.DataFrame({'Category': value_counts.index, 'Percentage': percentages })

   
    plost.donut_chart(
        data=df_plot,
        theta='Percentage',
        color='Category',
        legend='bottom'
    )

#Row C
st.subheader('Monthly Income vs Expenditure')

