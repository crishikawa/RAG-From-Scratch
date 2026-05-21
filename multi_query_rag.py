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
from langchain_core.load import dumps, loads
from operator import itemgetter

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

#### MULTI QUERY RETRIEVAL ####

# LLM
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Multi Query: 5 Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

# Retrieve
retrieval_chain = generate_queries | retriever.map() | get_unique_union

#RAG prompt
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain, "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

#### QUESTION LOOP ####

question = input("Ask a question (or 'quit' to exit): ").strip()
while question.lower() != "quit":
    result = final_rag_chain.invoke({"question":question})
    print("\n" + result + "\n")
    question = input("Ask another question (or 'quit' to exit): ").strip()
