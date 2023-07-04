import streamlit as st
from langchain.llms import OpenAI

st.title('🎈 App Name')

st.write('Ich bin eine Teständerung!')

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

openai_api_key = st.sidebar.text_input('OpenAI API Key')

#def generate_response(input_text):
#  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
#  st.info(llm(input_text))

#with st.form('my_form'):
#  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#  submitted = st.form_submit_button('Submit')
#  if not openai_api_key.startswith('sk-'):
#    st.warning('Please enter your OpenAI API key!', icon='⚠')
#  if submitted and openai_api_key.startswith('sk-'):
#    generate_response(text)

# import schema for chat messages and ChatOpenAI in order to query chatmodels GPT-3.5-turbo or GPT-4

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
     

chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
]
response=chat(messages)

print(response.content,end='\n')

# Import prompt and define PromptTemplate

from langchain import PromptTemplate

template = """
You are an expert data scientist with an expertise in building deep learning models. 
Explain the concept of {concept} in a couple of lines
"""

prompt = PromptTemplate(
    input_variables=["concept"],
    template=template,
)
     

# Run LLM with PromptTemplate

llm(prompt.format(concept="autoencoder"))
llm(prompt.format(concept="regularization"))

