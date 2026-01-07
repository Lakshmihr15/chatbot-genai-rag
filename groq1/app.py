import os
import warnings
import time
from dotenv import load_dotenv

# 1. Suppress the Pydantic/Python 3.14 warning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# 2. Set the User Agent for WebBaseLoader
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API Key
load_dotenv()
groq_api_key = os.getenv('groq_api_key')

# Streamlit Page Config
st.set_page_config(page_title="LangChain RAG Demo", page_icon="🤖")
st.header("🚀 Groq + LangChain RAG")

# 3. Vector Knowledge Base Initialization
if "vector" not in st.session_state:
    with st.status("🛠️ Building Knowledge Base...", expanded=True) as status:
        try:
            st.write("Connecting to Ollama (nomic-embed-text)...")
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            
            st.write("Fetching LangSmith Documentation...")
            st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
            docs = st.session_state.loader.load()
            
            st.write("Splitting content into optimized chunks...")
            # Smaller chunk size (700) is faster for i5 processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            final_documents = text_splitter.split_documents(docs)
            
            st.write("Creating FAISS Vector Store...")
            st.session_state.vector = FAISS.from_documents(final_documents, st.session_state.embeddings)
            
            status.update(label="✅ Ready!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            st.stop()

# 4. Chat Interface Logic
if groq_api_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based on the context only.
        <context>{context}</context>
        Question: {input}"""
    )

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_input = st.text_input("What would you like to know about LangSmith?")

    if user_input:
        start_time = time.process_time()
        with st.spinner("🤖 Searching documents and generating answer..."):
            response = retrieval_chain.invoke({"input": user_input})
        
        st.write(f"⏱️ Response time: {time.process_time() - start_time:.2f}s")
        st.markdown(f"### Answer:\n{response['answer']}")

        with st.expander("📚 Source Context"):
            for doc in response["context"]:
                st.caption(doc.page_content)
                st.divider()
else:
    st.error("🔑 Please check your .env file for the GROQ_API_KEY.")