import streamlit as st
from langchain.llms import OpenAI

st.title('Chatbot Example for Documents')

# st.write('Ich bin eine Teständerung!')

from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

# openai_api_key = st.sidebar.text_input('OpenAI API Key')

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
     

#chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
#messages = [
#    SystemMessage(content="You are an expert data scientist"),
#    HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
#]
#response=chat(messages)

#print(response.content,end='\n')

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
llm = OpenAI(temperature=0.7) # , openai_api_key=openai_api_key)
# llm(prompt.format(concept="autoencoder"))
# llm(prompt.format(concept="regularization"))

# Import LLMChain and define chain with language model and prompt as arguments.

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
# print(chain.run("autoencoder"))

# Define a second prompt 

second_prompt = PromptTemplate(
    input_variables=["ml_concept"],
    template="Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Define a sequential chain using the two chains above: the second chain takes the output of the first chain as input

from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# Run the chain specifying only the input variable for the first chain.
# explanation = overall_chain.run("autoencoder")
# print(explanation)

# Import utility for splitting up texts and split up the explanation given above into document chunks

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap  = 0,
# )

# texts = text_splitter.create_documents([explanation])

# st.write(texts)
# Import and instantiate OpenAI embeddings

from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

# loader = Docx2txtLoader("data/test.docx")
# st.write(loader)
# data = loader.load()
# st.write(data[0].page_content)

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# documents = text_splitter.split_documents(data)
# st.write(documents[0])

import os
from langchain.document_loaders import PyPDFLoader

documents = []

# st.write('Lese Dokumente ein...')

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        pdf_path = "data/" + file
        st.write('Verarbeite Datei: ' + pdf_path)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    # elif file.endswith('.docx') or file.endswith('.doc'):
    #     doc_path = "./docs/" + file
    #     loader = Docx2txtLoader(doc_path)
    #     documents.extend(loader.load())
    # elif file.endswith('.txt'):
    #     text_path = "./docs/" + file
    #     loader = TextLoader(text_path)
    #     documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)
# st.write(documents)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
     

# Turn the first text chunk into a vector with the embedding

#query_result = embeddings.embed_query(data[0].page_content)
#print(query_result)
#st.write(query_result)



# Do a simple vector similarity search

# query = "What is magical about an autoencoder?"
# result = search.similarity_search(query)

# print(result)
# st.write(result)

from langchain.vectorstores import Chroma

vectordb = Chroma.from_documents(
  documents,
  embedding=embeddings,
  persist_directory='./vector'
)
vectordb.persist()

from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI

# qa_chain = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
#     return_source_documents=True
# )

# we can now execute queries against our Q&A chain
# result = qa_chain({'query': 'Who is the CV about?'})
# print(result['result'])
# st.write(result['result'])

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    vectordb.as_retriever(search_type="mmr"), # search_kwargs={'k': 6}),
    return_source_documents=True
)

# import sys

chat_history = []
# chat_history_display = []

# add_selectbox = st.sidebar.selectbox(
#    "Historie",
#    chat_history_display
#)

with st.form('my_form'):
 text = st.text_area('Deine Frage', '...')
 submitted = st.form_submit_button('Fragen')
 if submitted:
    result = qa_chain({'question': text, 'chat_history': chat_history})
    st.write(result['answer'])
    # st.write(result)
    chat_history.append((text, result['answer']))
    st.write('Die Antwort basiert (unter anderem) auf diesem Text:')
    st.write(result['source_documents'][0].page_content)
    st.write('Datei: ' + result['source_documents'][0].metadata['source'])
    st.write('Seite: ' + str(result['source_documents'][0].metadata['page']))
    # st.write(type(result['source_documents'][0]))
    # st.write('Gefunden auf: ' + result['source_documents'][0].page)
    # st.write(chat_history)
    # print(chat_history)
    # chat_history_display.append('Frage: '+ text)

# while True:
#     # this prints to the terminal, and waits to accept an input from the user
#     query = input('Prompt: ')
#     # give us a way to exit the script
#     if query == "exit" or query == "quit" or query == "q":
#         print('Exiting')
#         sys.exit()
#     # we pass in the query to the LLM, and print out the response. As well as
#     # our query, the context of semantically relevant information from our
#     # vector store will be passed in, as well as list of our chat history
#     result = qa_chain({'question': query, 'chat_history': chat_history})
#     print('Answer: ' + result['answer'])
#     # we build up the chat_history list, based on our question and response
#     # from the LLM, and the script then returns to the start of the loop
#     # and is again ready to accept user input.
#     chat_history.append((query, result['answer']))

# Import Python REPL tool and instantiate Python agent

# from langchain.agents.agent_toolkits import create_python_agent
# from langchain.tools.python.tool import PythonREPLTool
# from langchain.python import PythonREPL
# from langchain.llms.openai import OpenAI

# agent_executor = create_python_agent(
#     llm=OpenAI(temperature=0, max_tokens=1000),
#     tool=PythonREPLTool(),
#     verbose=True
# )
     

# Execute the Python agent

# test = agent_executor.run("Find the roots (zeros) if the quadratic function 3 * x**2 + 2*x -1")
# st.write(test)
