import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

# Modern LangChain & LangServe imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langserve import add_routes

# 1. Load environment variables (for LangSmith tracing if you use it)
load_dotenv()

# 2. Initialize FastAPI App
app = FastAPI(
    title="Gemma API Server",
    version="1.0",
    description="A local API server for Gemma using Ollama"
)

# 3. Initialize Local Gemma Model
# This connects to your local Ollama instance (no API key needed)
llm = OllamaLLM(model="gemma")

# 4. Define the Prompt Template
prompt = ChatPromptTemplate.from_template("Write a short essay about {topic} with about 100 words.")

# 5. Add the Route
# This creates the /essay/invoke and /essay/playground endpoints
add_routes(
    app,
    prompt | llm,
    path="/essay"
)

# 6. Run the Server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)