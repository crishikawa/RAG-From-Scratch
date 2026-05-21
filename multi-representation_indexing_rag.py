import os
from dotenv import load_dotenv
load_dotenv()

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
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

# LLM
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Generate a summary for each chunk
summary_chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document chunk concisely:\n\n{doc}")
    | llm
    | StrOutputParser()
)

print("\nGenerating summaries for indexing...")
summaries = summary_chain.batch(splits, {"max_concurrency": 3})

#### MULTI-REPRESENTATION SETUP ####

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generate unique IDs to link summaries to their original chunks
doc_ids = [str(uuid.uuid4()) for _ in splits]

# Map each ID to its corresponding full original chunk
docstore = dict(zip(doc_ids, splits))

# Vectorstore indexes summaries with ID metadata
summary_docs = [
    Document(page_content=s, metadata={"doc_id": doc_ids[i]})
    for i, s in enumerate(summaries)
]

vectorstore = Chroma.from_documents(
    documents=summary_docs,
    embedding=embeddings,
    collection_name="summaries"
)

print("Indexing complete — summaries indexed, full chunks stored.\n")

#### RETRIEVAL AND GENERATION ####

rag_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(rag_template)

def retrieve_full_docs(question, k=4):
    """Search summaries, return full original chunks"""
    # Search vectorstore using summaries
    results = vectorstore.similarity_search(question, k=k)
    # Look up full chunks using the doc_id from metadata
    full_docs = [docstore[r.metadata["doc_id"]] for r in results]
    return full_docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#### QUESTION LOOP ####

question = input("Ask a question (or 'quit' to exit): ").strip()
while question.lower() != "quit":
    # Search uses summaries, but Claude gets full chunks
    retrieved_docs = retrieve_full_docs(question)
    context = format_docs(retrieved_docs)
    result = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    print("\n" + result + "\n")
    question = input("Ask another question (or 'quit' to exit): ").strip()