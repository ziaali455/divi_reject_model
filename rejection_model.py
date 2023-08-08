import pickle
import streamlit as st
st.markdown("test")
loaded_model = pickle.load(open("model.pkl", 'rb'))
