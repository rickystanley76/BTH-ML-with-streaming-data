# BTH-ML-with-streaming-data [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rickystanley76/bth-ml-with-streaming-data/main/app.py)
The project is about SteamFlow prediction of a paper mil. The target is to predict how much water needed to make a paper roll.

# app.py
Is the main file to run for the app to see the result. Streamlit library was used to make the app.

# papermil_EDA_Testing_with_different_algorithms.ipynb
Where EDA was done and applied different algorithm to find our which gives the minimum error. 
   Algorithms	            R2	     RMSE
1.	Linear Regression	0.68237	0.75152
2.	XGBoost	0.80377	0.562834
3.	Random Forest with HT	0.7862	0.61646
4.	Decision Tree with HT	0.51144	0.93205


(HT means hyperparmeter tunig)

We found Random forest gives the lowest error, so we used that to predict the steamflow in the app.py 

# papermil_xgb.pkl
The pickel file is created from Notebook file using the XGBoost Algorithm. 

# papermil_water_flow.csv
The data of the paper mil, 11 features and 1(steamflow) target. 




