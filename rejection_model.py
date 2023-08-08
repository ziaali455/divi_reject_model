import pickle
import streamlit as st
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
st.markdown("test")
loaded_model = pickle.load(open("model.pkl", 'rb'))
raw_data = {'name': ['Bobs Burgers'], 'age': [3,], 'growth_overall': [-24.56]}  
#inputs: 
data = pd.DataFrame(raw_data)
data_scaled = scaler.fit_transform(data.drop(columns=['name']))
st.markdown(loaded_model.predict(data_scaled))