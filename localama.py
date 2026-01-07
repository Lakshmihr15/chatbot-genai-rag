import streamlit as st
import os
from dotenv import load_dotenv

# Updated imports for modern LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title("LangChain Demo with Ollama Gemma")
input_text = st.text_input("Search the topic you want:")

# Initialize LLM (using the gemma model you verified earlier)
llm = OllamaLLM(model="gemma")
output_parser = StrOutputParser()

# Create chain using the pipe operator
chain = prompt | llm | output_parser

# Run the chain when user enters input
if input_text:
    with st.spinner("Gemma is thinking..."):
        # Use .invoke() instead of .run()
        response = chain.invoke({"question": input_text})
        st.write(response)