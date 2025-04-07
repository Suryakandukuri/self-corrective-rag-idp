from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import chromadb
from langchain_chroma import Chroma
from utils.data_gatherer import fetch_ckan_package_data

load_dotenv()

class RAG:
    def __init__(self) -> None:
        self.emb_model_path = "BAAI/bge-base-en-v1.5"
        self.emb_model = self.get_embedding_model(self.emb_model_path)
        self.vector_store_path = os.getenv('VECTOR_STORE')
        self.chroma_client = chromadb.PersistentClient(path=self.vector_store_path)
        self.collection_name = "idp_search"

    def get_embedding_model(self, emb_model_path):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceBgeEmbeddings(
            model_name=emb_model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def split_docs(self, docs):
        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=0)
        return text_splitter.split_documents(docs)

    def populate_vector_db(self):
        documents = fetch_ckan_package_data()
        
            # Delete existing collection
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass
        # Process and add documents to ChromaDB
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(
                Document(
                    page_content=doc['text'],
                    metadata=doc['metadata']
                )
            )
        # Use LangChain's Chroma to create the collection with the correct embedding function
        vectorstore = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.emb_model,
            persist_directory=self.vector_store_path,
            collection_name=self.collection_name
        )

        return vectorstore

    def load_vector_db(self):

        return Chroma(
            persist_directory = self.vector_store_path,
            # client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.emb_model
        )

    def get_retriever(self):
        return self.load_vector_db().as_retriever(search_type="mmr",
                search_kwargs={'k': 10, 'lambda_mult': 0.25})
