import langchain_helper as lch 
import streamlit as st
import textwrap
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Adjusting the title width using markdown and custom CSS
st.markdown(
    """
    <style>
    .title {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        width: 100%;
    }
    .subtitle {
        font-size: 20px;
        font-weight: normal;
        text-align: center;
        width: 100%;
        color: gray;
    }
    .response {
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="title">Chatting with Obsolete Software Requirements by Wnuk, Gorschek and Zahda</div>
    <div class="subtitle">SWEN 645 9041 Software Requirements</div>
    """,
    unsafe_allow_html=True
)

# Load the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please set it in the environment variables.")

# Load the PDF data and create the vector store at the start
documents = lch.load_pdf_data("docs2")
if not documents:
    raise ValueError("No documents were loaded. Please check the PDF directory path and files.")
vector_store = lch.create_vector_store(documents, openai_api_key=openai_api_key)

with st.form(key='my_form'):
    query = st.text_area(label='Please enter your question', key='query')
    submitted = st.form_submit_button(label='Submit')

    if submitted:
        if query == "":
            st.write("Please enter a question.")
            st.stop()
        st.write("Processing...")
        try:
            # Validate the API key
            lch.validate_openai_api_key(openai_api_key)
            
            response = lch.get_response_from_query(vector_store, query, openai_api_key=openai_api_key)
            st.subheader("Response:")
            st.markdown(f'<div class="response">{textwrap.fill(response, width=100)}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.write("An error occurred. Please try again.")
            st.write(e)