import pickle
import streamlit as st
import pandas as pd

st.markdown("# Divibank Summer 2023 Rejection Model Demo - Ali Zia")
st.markdown("## Rejection Model:")
loaded_rejection_model = pickle.load(open("model.pkl", 'rb'))
company_name = st.text_input('Client Name')
company_age = st.text_input('Company Age (Years)')
company_growth = st.text_input('Overall Growth (%)')

if st.button("Generate Result #1"):
    raw_data = {'name': [company_name], 'age': [int(company_age)], 'growth_overall': [float(company_growth)]}  
    data = pd.DataFrame(raw_data)
    data_scaled = scaler.fit_transform(data.drop(columns=['name']))
    st.markdown("### Approved?: ")
    st.markdown(loaded_rejection_model.predict(data_scaled))

st.markdown("# Divibank Summer 2023 Loan Plan Model Demo - Ali Zia")
st.markdown("## Loan Model (Interest Rate):")
loaded_loan_model = pickle.load(open("prelim_data.pkl", 'rb'))
name = st.text_input('Client Name',key="1")
age = st.text_input('Company Age (Years)',key="11")
gross = st.text_input('Gross Revenue',key="111")
rating = st.text_input('Model Rating (AAA-D)',key="1111")

def rating_to_int(full_client_data, rating):
  ratings = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "B": 7.5,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB": 10,
    "CCC+": 11,
    "CCC": 12,
    "CCC-": 13,
    "CC": 14,
    "C": 15,
    "D": 16,
  }
  values = {rating: 'CC'}
  full_client_data.fillna(value=values)

  full_client_data[rating] = full_client_data[rating].map(ratings)
  return full_client_data

def dummy_ratings(full_client_data):
  df_dummies = full_client_data.join(pd.get_dummies(full_client_data["Model Rating"], prefix="model_rating"))
  print("Dummies: ", df_dummies.shape)
  print("full_client_data: ", full_client_data.shape)
  return df_dummies


def eval_dummies(data):
    return loaded_loan_model.predict(pd.DataFrame.from_dict(data))

if st.button("Generate Result #2"):
   raw_data = {'Age (Years)': [age], '01_Gross_revenue': [gross], 'Model Rating': [rating]}
   raw_data = pd.DataFrame.from_dict(raw_data)
   raw_data = rating_to_int(raw_data, 'Model Rating')
   st.markdown(raw_data)
   st.markdown("### Interest Rate: ")
   st.markdown(eval_dummies(dummy_ratings(raw_data)))




