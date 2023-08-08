import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
st.markdown("# Divibank Summer 2023 Model Demo - Ali Zia")
st.markdown("## Rejection Model:")
loaded_model = pickle.load(open("model.pkl", 'rb'))
company_name = st.text_input('Client Name')
company_age = st.text_input('Company Age (Years)')
company_growth = st.text_input('Overall Growth (%)')

#inputs: 
if st.button("Generate Result"):
    raw_data = {'name': [company_name], 'age': [int(company_age)], 'growth_overall': [float(company_growth)]}  
    data = pd.DataFrame(raw_data)
    data_scaled = scaler.fit_transform(data.drop(columns=['name']))
    st.markdown("### Approved?: ")
    # if(loaded_model.predict(data_scaled)==False):
    #     st.markdown("Not Approved")
    # else:
    #     st.markdown("Approved")
    st.markdown(loaded_model.predict(data_scaled))


