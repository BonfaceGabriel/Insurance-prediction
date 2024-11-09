# Insurance-prediction

### Objective
This project aims to predict which clusters in Nakuru County would likely buy which insurance products. <br>
Its main objective is to aid insurance companies tailor their product marketing and advertising appropriately to groups most likely to buy them, increasing the probability of a sale.

### Dataset
The dataset used for this project is the Kenya Financial Access Dataset 2021 from the Kenya National Bureau of Statistics.<br> The dataset is very big containing over 2000 columns.<br> To ensure only reliable attributes are used, the main dataframe used for this project is a sample from the original data.<br>
Origninal Dataset: https://docs.google.com/spreadsheets/d/1IZ-6HCDzeypoa4IQNr-nH0Y26Qc5EwSE/edit?usp=sharing&ouid=115471613002921533452&rtpof=true&sd=true

### Notebook
Utilizes Kprototypes Clustering model to divide the data into clusters.
There are cluster profiles and the probable insurance products those clusters may purchase at the end of the notebook.

### APP
Used for the prediction and insurance product recommendation.<br>
Features:
- Shows shap summary plots for different clusters
- Makes cluster predictions based on user inputs
- Makes insurance product recommendation based on the different clusters
  
##### Install
- Fork the repository
- Run *pip install -r requirements.txt* to install required dependencies
- Run *streamlit run app.py*

##### Navigation
The App has three pages:

App:
- Sidebar where the user uploads a CSV or manually inputs data
- User input is shown after app introduction.
- BUtton to make the prediction and recommendation.
- Dashboard for various visualizations
- Profile for general information on data used

Profile:
- Shows data profile

Dashboard:
- Side bar has user input parameters for some of the charts
- Shows visualizations of the data

