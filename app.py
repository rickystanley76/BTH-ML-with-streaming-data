import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import shap
import xgboost
import matplotlib.pyplot as plt
import time


def run_status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Percent Complete {i+1}')
		bar.progress(i + 1)
		time.sleep(0.1)
		st.empty()


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('SteamFlow Prediction APP for a Papermil ')
from PIL import Image
image = Image.open('papermil1000.jpg')
st.image(image, caption='Paper mill- Paper production (Own image)', width=400)

# EDA
st.subheader('Exploratory Data Analysis')
my_dataset = "papermil_water_flow.csv"

# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

def user_input_features():
   
        pulp_to_mixing_tank23 = st.sidebar.slider('Pulp to mixing TANK23', 131.0,223.0,175.0,0.2)
        flow_after_machine_chest21 = st.sidebar.slider('Flow after Machine_Chest21', 2148.0,3700.0,2888.0,0.5)
        level_condensate_bucket1_valve_position = st.sidebar.slider('level condensate bucket1 valve position', 26.0,531.0,330.0,0.2)    
        steam_group3_pressure = st.sidebar.slider('Steam group3 pressure', 26.0,531.0,330.0,0.2)                
        steam_group4under_pressure = st.sidebar.slider('Steam group4 under_pressure', 26.0,531.0,330.0,0.2)                      
        pressure_yankee_cylinder = st.sidebar.slider('Pressure of yankee cylinder', 143.0,295.0,237.0,0.2)
        steam_group5under_pressure = st.sidebar.slider('Steam Group5under Pressure', 143.0,287.0,237.0,0.2)       
        steam_pressure5_over = st.sidebar.slider('Steam pressure5 over', 143.0,290.0,237.0,0.2)
        pressure_condensate_bucket5 = st.sidebar.slider('Pressure_condensate_bucket5', 2.0,100.0,75.0,0.2)
        production_paper_machine2 = st.sidebar.slider('Production PAPERMACHINE2', 0.0,14.0,10.0,0.2)
        dry_production_vira = st.sidebar.slider('Dry production Vira', 115.0,205.0,160.0,0.2)
   
       
       
        
        data = {'pulp_to_mixing_tank23': pulp_to_mixing_tank23,             
                'flow_after_machine_chest21': flow_after_machine_chest21,           
                'level_condensate_bucket1_valve_position': level_condensate_bucket1_valve_position,
                'steam_group3_pressure': steam_group3_pressure,
                'steam_group4under_pressure': steam_group4under_pressure,             
                'pressure_yankee_cylinder': pressure_yankee_cylinder,
                'steam_group5under_pressure': steam_group5under_pressure,
                'steam_pressure5_over': steam_pressure5_over,            
                'pressure_condensate_bucket5': pressure_condensate_bucket5,
                'production_paper_machine2': production_paper_machine2,
                'dry_production_vira': dry_production_vira           
               }
        features = pd.DataFrame(data, index=[0])
        return features
   


# Show Entire Dataframe
if st.checkbox("Show Dataset used in the Model Building"):
	data = explore_data(my_dataset)
	st.dataframe(data)

# Show Description
if st.checkbox("Show All Column Names"):
	data = explore_data(my_dataset)
	st.text("Columns:")
	st.write(data.columns)
    
# Dimensions
data_dim = st.radio('What Dimension Do You Want to Show',('Rows','Columns'))
if data_dim == 'Rows':
	data = explore_data(my_dataset)
	st.text("Showing Length of Rows")
	st.write(len(data))
if data_dim == 'Columns':
	data = explore_data(my_dataset)
	st.text("Showing Length of Columns")
	st.write(data.shape[1])


if st.checkbox("Show Summary of Dataset"):
	data = explore_data(my_dataset)
	st.write(data.describe())
    

showlinechart= st.checkbox('Show line chart for SteamFlow')   
    
if showlinechart:
    st.line_chart(data['steamflow'])



showdescription = st.checkbox('Show Project Description')

if showdescription:
    st.write("""
# Description

This app predicts the **SteamFlow** of a PaperRoll in PaperMill!

During the production of a paper roll in the paper mill, it takes around **17-20** tons of water/hour(Depends on the size of the roll).
There are 403 parameters nobs are all around the paper machine, which measures different parameters during the production. Among them, 11 are
more important that has good effect on the water uses(SteamFlow) parameter.

With this app production manager can check how much water will be used to produce a paper roll and set the parameters to those 12 nobs.

#  Tools used:

We have used XGBoost with the following parameters:
    XGBRegressor(colsample_bytree=0.7949874768919643, gamma=0.4610218094027685,
             learning_rate=0.34903254668281886, max_depth=13,
             min_child_weight=53.33039107887038, n_estimators=1235, nthreads=-1,
             reg_alpha=9.683328077024195, subsample=0.9849758415841514)

Data we had collected from a paper mill open data. The data consists of 3941 records with 11(independent) 1(dependent) features. 

Used Streamlit Library.

""")

st.sidebar.header('User Input Features- 11 parameters: Change the parameters and see the result in the right side')


input_df = user_input_features()
#as Random forest gives us the minimum error(RMSE), we will use that to 
#find the prediction of Steamflow


# Reads in saved XGB  model
load_xgb_model = pickle.load(open('papermil_xgb.pkl', 'rb'))

# Apply XGB model to make predictions
prediction_xgb = load_xgb_model.predict(input_df)

# # Reads in saved Orange AdaBoost  model
# load_ada_model = pickle.load(open('Orange_adaboost_Model.pkcls', 'rb'))


# # Apply Adaboost model to make predictions
# prediction_ada = load_ada_model.predict(input_df)

## Result displaying in Table
resultframe = {'XGBoost':  [prediction_xgb]
        #'AdaBoost': [prediction_ada]
        }

df_res = pd.DataFrame (resultframe, columns = ['XGBoost'])

st.write("""
         # Below is the Result :(Tons/hour)""")
st.write(df_res)


#Model Explainability--

st.write("""
         # Model Explainability- XGBoost """)

st.subheader("""
# Why it is important?""")

st.write("""
Machine learning can't be a black box today, as GDPR implemented, model should have the power to explain.
By which companies can built trust among user and customer using a machine learning model.

""")         
         
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap

explainer = shap.TreeExplainer(load_xgb_model)
shap_values = explainer.shap_values(input_df)



st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
st.text('Explaining the model predictions using SHAP values')
st.write("SHAP in Github [link](https://github.com/slundberg/shap)")
run_status()
shap.summary_plot(shap_values, input_df)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, input_df, plot_type="bar")
st.pyplot(bbox_inches='tight')



st.subheader("Model-Built with XGBoost, Deployed with Streamlit")
st.text("by: Ricky D'Cruze")
st.write("Source code, Data, pickle file,notebook in Github [link](https://github.com/rickystanley76/BTH-ML-with-streaming-data)")   


