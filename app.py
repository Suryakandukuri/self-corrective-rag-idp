
#%%
# import libraries
from fastapi import FastAPI, Query
from utils.build_rag import RAG
from utils.llm import LLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# define LLM
llm = LLM()
groq = llm.get_groq_llm()

# load vector DB
rag = RAG()

rag.load_vector_db()

retriever = rag.get_retriever()

# Grading and Self Corrective Mechanism with retrieval_grader

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = groq.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
# %%
question = "electricity"

docs = retriever.invoke(question)

for doc in docs:
    doc_text = doc.page_content
    print(doc_text)
# doc_txt = docs[1].page_content
# print(doc_txt)

# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
# %%
