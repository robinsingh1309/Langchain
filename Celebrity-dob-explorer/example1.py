import os
import warnings

#import the api key from the constants.py file
from constants import groq_api_key

# which llm to use
from langchain_groq import ChatGroq

# to write the prompt templates
from langchain.prompts import PromptTemplate

# to feed the prompt templates to the llms
from langchain.chains import LLMChain

# to run the multiple chains in a sequence
from langchain.chains import SequentialChain

# for storing the conversation and save in the memory
from langchain.memory import ConversationBufferMemory

# use streamlit to create a web app with UI
import streamlit as st


 # Create custom UI on streamlit
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🌟 Celebrity Birth Date Explorer
</h1>

<h4 style='text-align: center; color: gray;'>
Learn about your favorite celebrity and discover important world events around their birth.
</h4>
""", unsafe_allow_html=True)

st.divider()

input_text = st.text_input(
    "🔎 Enter a celebrity name",
    placeholder="e.g. Cristiano Ronaldo, Emma Watson"
)

# ChatGroq LLMS
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8, api_key=groq_api_key)



# Memory to store the conversation
person_memory = ConversationBufferMemory(input_key="name", memory_key="name_history")
dob_memory = ConversationBufferMemory(input_key="person", memory_key="person_history")
description_memory = ConversationBufferMemory(input_key="birth_date", memory_key="description_history")



# First Prompt Template
first_input_prompt = PromptTemplate(
        input_variables=["name"], 
        template="Tell me about celebrity {name} under the 60 words."
    ) 

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="person", memory=person_memory) # verbose is a parameter that controls whether the model's output is printed to the console.



# Second Prompt Template
second_input_prompt = PromptTemplate(
        input_variables=["person"], 
        template="When was {person} born? Respond only with the date in the format for example: 1 January 2000."
    ) 

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="birth_date", memory=dob_memory)



# Third Prompt Template
third_input_prompt = PromptTemplate(
        input_variables=["birth_date"], 
        template="List exactly five major world events that happened around {birth_date}. Use one short sentence per event and list them in bullet points."
    ) 

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="description", memory=description_memory)


# Combine the chains
parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables=["name"], output_variables=["person", "birth_date", "description"], verbose=True)


# running the streamlit app 
if input_text:
    st.write(parent_chain({"name": input_text}))

    with st.expander("Person Name"):
        st.info(person_memory.buffer)
    
    with st.expander("Major Events"):
        st.info(description_memory.buffer) 