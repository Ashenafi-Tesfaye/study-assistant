import langchain_helper as lch 
import streamlit as st
import textwrap

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

with st.form(key='my_form'):
     # Input field for the OpenAI API key
    openai_api_key = st.text_input("Please enter your OpenAI API key", type="password")
    
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
            
            response = lch.get_response_from_query(lch.vector_store, query)
            st.subheader("Response:")
            st.markdown(f'<div class="response">{textwrap.fill(response, width=100)}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.write("An error occurred. Please try again.")
            st.write(e)
