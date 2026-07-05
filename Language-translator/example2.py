import os

# import the api key from the constants.py file
from constants import groq_api_key

# which llm to use
from langchain_groq import ChatGroq

# import the PromptTemplate class from langchain.prompts
from langchain.prompts import PromptTemplate

# use the LLMChain class from langchain.chains
from langchain.chains import LLMChain

# use the Streamlit library for creating the web app
import streamlit as st


# ChatGroq LLM
llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8, api_key=groq_api_key)


st.title("Language translation")
input_text = st.text_input("Enter text to translate")



first_input_prompt = PromptTemplate(
        input_variables=["text", "target_language"], 
        template="Translate the following text to {target_language}: {text} in one line with no suggestion or explanation.")

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="translation")


languageToTranslate = st.text_input("Enter target language")

if input_text and languageToTranslate:
    st.write(chain.run(text=input_text, target_language=languageToTranslate))