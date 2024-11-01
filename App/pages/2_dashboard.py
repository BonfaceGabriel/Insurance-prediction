import streamlit as st
import pandas as pd
import plost
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards
import seaborn as sns
sns.set()

data = pd.read_csv('../Nakuru_FinAccess1.csv')

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

st.sidebar.subheader('Chart parameters')
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

st.title('DASHBOARD')

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
c1, c2 = st.columns((5, 4))
with c1:
    st.subheader('Age Distribution')
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(data['age_of_respondent'], edgecolor='black')
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
c1, c2 = st.columns((6, 4))
with c1:
    st.subheader('Monthly Income vs Expenditure')
    st.scatter_chart(data, x='avg_mnth_income', y='total_exp_per_moth', height=plot_height)
with c2:
    st.subheader('Use of insurance by Area')
    sns.countplot(x='area', hue='insurance', data=data)
    plt.xlabel('Area')
    plt.ylabel('Count')
    st.pyplot(plt)

#Row D
c1, c2 = st.columns(2)
with c1:
    st.subheader('Insurance use by Average Monthly Income')
    data['income_bins'] = pd.cut(data['avg_mnth_income'], bins=[0, 10000, 20000, 50000, data['avg_mnth_income'].max()],
                                labels=['Low Income', 'Middle Income', 'High Income', 'Very High Income'])

    # Bar plot for insurance usage across income bins
    plt.figure(figsize=(10, 5))
    sns.countplot(x='income_bins', hue='insurance', data=data)
    plt.xlabel('Income Level')
    plt.ylabel('Count')
    st.pyplot(plt)

with c2:
    st.subheader('Insurance Use by Attribute')
    plt.figure(figsize=(10, 5))
    sns.countplot(x=attribute, hue='insurance', data=data)
    if attribute == 'nearest_financial_prod':
        plt.xlabel('Nearest Financial Product')
    elif attribute == 'most_important_life_goal':
        plt.xlabel('Most Important Life Goal')
    else:
        plt.xlabel('Source of Income')
    
    plt.ylabel('Count')
    st.pyplot(plt)
    

