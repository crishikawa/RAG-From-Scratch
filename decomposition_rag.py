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

#### DECOMPOSITION ####

# LLM
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Break down the input question into 3 sub-questions
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# Generate sub-questions chain
generate_queries_decomposition = ( 
    prompt_decomposition 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# RAG prompt for each sub-question
sub_question_template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt_rag = ChatPromptTemplate.from_template(sub_question_template)

# Combines all sub-answers into a final answer
synthesis_template = """Here is a set of sub-questions and their answers:

{context}

Use these to synthesize a final answer to the original question: {question}
"""
prompt_synthesis = ChatPromptTemplate.from_template(synthesis_template)

def answer_with_decomposition(question):
    """Break question into sub-questions, answer each, then synthesize"""

    # Generate sub-questions
    sub_questions = generate_queries_decomposition.invoke({"question": question})

    # Answer each sub-question individually
    qa_pairs = ""
    for i, sub_q in enumerate(sub_questions, 1):
        retrieved_docs = retriever.invoke(sub_q)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        answer = (prompt_rag | llm | StrOutputParser()).invoke({
            "context": context,
            "question": sub_q
        })
        qa_pairs += f"Sub-question {i}: {sub_q}\nAnswer: {answer}\n\n"

    # Synthesize final answer
    final_answer = (prompt_synthesis | llm | StrOutputParser()).invoke({
        "context": qa_pairs,
        "question": question
    })

    return final_answer

#### QUESTION LOOP ####

question = input("Ask a question (or 'quit' to exit): ").strip()
while question.lower() != "quit":
    result = answer_with_decomposition(question)
    print("\n" + result + "\n")
    question = input("Ask another question (or 'quit' to exit): ").strip()
