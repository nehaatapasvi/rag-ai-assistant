import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm=Ollama(model="llama3")

prompt=ChatPromptTemplate.from_template("You are a helpful assistant who tells funny jokes,Tell me one joke about{topic}")

parser=StrOutputParser()

chain=prompt|llm|parser

st.title("joke generator raa buahahaha🤠")

topic=st.text_input("enter a topic you want joke about")

# print("give me the topic you want joke about")

# input=input()

# response=chain.invoke({"topic":input})

# print("Here is your joke enjayy")
# print(response)

if(st.button("Get joke")):
    if topic.strip()=="":
        st.warning("please enter a topic")
    else:
        with st.spinner("generating your joke"):
            response=chain.invoke({"topic":topic})
        st.success("Here is your joke:")
        st.write(response)

