import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

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
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embeddings)

retriever = vectorstore.as_retriever()

#### SEMANTIC ROUTING ####

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Define prompt templates — router picks the most semantically similar one
beginner_template = """simple explanation easy understand basic introduction \
fundamental concepts everyday examples no jargon beginners students learning \
curious newcomers straightforward clear accessible

Here is a question:
{query}"""

advanced_template = """technical advanced expert precise methodology theoretical \
academic rigorous analytical complex specialized professional research scholarly \
nuanced sophisticated in-depth

Here is a question:
{query}"""

# Embed the prompt templates so we can compare questions against them
prompt_templates = [beginner_template, advanced_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to the most similar prompt template
def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    chosen = "BEGINNER" if most_similar == beginner_template else "ADVANCED"
    print(f"\n[Routed to: {chosen}]")
    
    # Retrieve docs and inject into the prompt
    retrieved_docs = retriever.invoke(input["query"])
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Add context to the template
    template_with_context = most_similar + f"\n\nContext:\n{context}"
    return PromptTemplate.from_template(template_with_context)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)

#### QUESTION LOOP ####

question = input("Ask a question (or 'quit' to exit): ").strip()
while question.lower() != "quit":
    result = chain.invoke(question)
    print("\n" + result + "\n")
    question = input("Ask another question (or 'quit' to exit): ").strip()