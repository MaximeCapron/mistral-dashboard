import streamlit as st
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
import pandas as pd

# Load the images
mistral_path = 'announcing-mistral.png'
mistral_logo = open(mistral_path, 'rb').read()
bnp_path = 'bnp_logo_white.png'
bnp_logo = open(bnp_path, 'rb').read()

# Define a function to call the Mistral models
def mistral(message, model):
    api_key = '0Hwnas53O1wmHPOZldhlwhqbh381NFIN'
    client = MistralClient(api_key=api_key)
    chat_stream = client.chat_stream(
        model=model,
        messages=[ChatMessage(role='user', content=message)]
    )
    for chunk in chat_stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            yield delta

# Set up the Streamlit app
st.set_page_config(page_title="Mistral FinNemo", layout="wide")

# Create columns
col1, col2, col3 = st.columns([15,3,2])

# Display the images at the top right of the page
with col1:
    st.title("Testing Mistral FinNemo")

with col2:
    st.write("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.image(bnp_logo)  

with col3:
    st.image(mistral_logo)  

# User input
user_message = st.text_input("Type a question:")

# Button to trigger the model response
if st.button("Enter"):
    if user_message:
        with st.spinner("Fetching responses..."):
            left_response = mistral(user_message, 'open-mistral-nemo')
            right_response = mistral(user_message, 'ft:open-mistral-nemo:8fe421f9:20240726:9e9083fe')
        
        # Display the responses
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Mistral Nemo Model")
            st.write(left_response)

        with col2:
            st.subheader("Fine-tuned Nemo Model for Finance")
            st.write(right_response)

    else:
        st.warning("Please enter a question.")
