import streamlit as st

st.title("Streamlit Test App")
st.write("Hello from Streamlit!")

name = st.text_input("Enter your name", "World")
if st.button("Greet"):
    st.success(f"Hello, {name}!")

st.slider("Pick a number", 0, 100, 25)
