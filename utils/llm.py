import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
class LLM:
    def __init__(self) -> None:
        load_dotenv()
        os.environ["GROQ_API_KEY"] = os.environ['GROQ_API_KEY']

    def get_embeddings(self):
        """Open-source alternative without Azure dependencies"""
        return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    def get_groq_llm(self):
        """
        Creates a ChatGroq instance for the LLaMA 3 70B 8192 model using the GROQ API key
        stored in the GROQ_API_KEY environment variable.

        Returns:
            ChatGroq
        """
        return ChatGroq(
            groq_api_key=os.environ['GROQ_API_KEY'],
            model_name="llama3-70b-8192"
        )