import requests
import streamlit as st

def get_Ollama_response(input_text):
    try:
        # Note the nested "input" dictionary below
        response = requests.post(
            "http://localhost:8000/essay/invoke",
            json={'input': {'topic': input_text}}
        )
        
        # Access the 'output' from the JSON response
        return response.json()['output']
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the server. Is app.py running?"
st.title("langchain demo with gemma API")
input_text=st.text_input("write essay on")

if input_text:
    st.write(get_Ollama_response(input_text))

    