from utils.build_rag import RAG

# initiate RAG and execute RAG
rag = RAG()

# gathering the data
docs = rag.populate_vector_db()

# getting the retriever
retriever = rag.get_retriever()
