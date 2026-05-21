import os
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

#### INDEXING ####

# Load Documents
source = input("Enter a URL or path to a PDF: ")

if source.endswith(".pdf"):
    loader = PyPDFLoader(source)
else:
    loader = WebBaseLoader(web_paths=(source,))

docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

retriever = vectorstore.as_retriever()

#### HYDE RETRIEVAL ####

# LLM
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# HyDE document generation
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

generate_docs_for_retrieval = (
    prompt_hyde
    | llm
    | StrOutputParser() 
)

# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever

#RAG prompt
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

#### QUESTION LOOP ####

question = input("Ask a question (or 'quit' to exit): ").strip()
while question.lower() != "quit":
    retrieved_docs = retrieval_chain.invoke(question)
    result = final_rag_chain.invoke({"context":retrieved_docs,"question":question})
    print("\n" + result + "\n")
    question = input("Ask another question (or 'quit' to exit): ").strip()
