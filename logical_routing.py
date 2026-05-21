import os
from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

#### INDEXING ####

# Load two sources
print("Load two sources\n")

def load_source(label):
    source = input(f"Enter URL or PDF path for source {label}: ").strip()
    if source.endswith(".pdf"):
        loader = PyPDFLoader(source)
    else:
        loader = WebBaseLoader(web_paths=(source,))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        collection_name=f"source_{label}"
    )
    return vectorstore.as_retriever(), source

retriever_a, source_a = load_source("A")
retriever_b, source_b = load_source("B")

#### ROUTING ####

# LLM
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["source_a", "source_b"] = Field(
        ...,
        description="Given a user question choose which datasource is most relevant"
    )

structured_llm = llm.with_structured_output(RouteQuery)

# Router prompt
router_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are an expert at routing questions to the right datasource.
You have two sources available:
- source_a: {source_a}
- source_b: {source_b}

Route the question to whichever source is most likely to contain the answer."""),
    ("human", "{question}"),
])

router = router_prompt | structured_llm

# RAG prompt
rag_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

def route_and_answer(question):
    """Route question to correct source then run RAG"""

    # Decide which source to use
    route = router.invoke({"question": question})
    retriever = retriever_a if route.datasource == "source_a" else retriever_b
    source_used = source_a if route.datasource == "source_a" else source_b

    print(f"\n[Routed to: {source_used}]")

    # Retrieve and answer
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    answer = (rag_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    return answer

#### QUESTION LOOP ####

question = input("\nAsk a question (or 'quit' to exit): ").strip()
while question.lower() != "quit":
    result = route_and_answer(question)
    print("\n" + result + "\n")
    question = input("Ask another question (or 'quit' to exit): ").strip()