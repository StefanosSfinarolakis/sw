import streamlit as st
from data_loader import load_data

st.title("Data Analysis Application")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write(data)