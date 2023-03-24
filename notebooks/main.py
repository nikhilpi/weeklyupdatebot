import streamlit as st
from WeeklyUpdateAgent import Weeklupdate

# streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false

agent = ''

@st.cache_resource
def load_agent():
    if agent:
        return agent
    else:
        return Weeklupdate()

st.header("Ask Anthony Anything")
def get_text():
    input_text = st.text_area(label="Question", label_visibility='collapsed', placeholder="Your question...", key="question_input")
    return input_text

question_input = get_text()

if question_input:
    st.write("Your question was: ", question_input)
    
if question_input:

    output = load_agent().query_agent(question_input)
    st.write(output)

if st.button('Reload DB'):
    load_agent().setup_db()