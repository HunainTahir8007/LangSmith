import os
from dotenv import load_dotenv
from langsmith import traceable  
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


os.environ["LANGSMITH_PROJECT"] = "MY RAG CHATBOT"
load_dotenv()

file_PATH = ""  # provide the path


model =  ChatGroq(model = "llama-3.3-70b-versatile" , temperature= 0.2)

@traceable(name="loading_file")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load() 

@traceable(name="document_split_block")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="vectorstore_creation")
def build_vectorstore(splits):
    emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    vs = FAISS.from_documents(splits, emb)
    return vs

@traceable(name="full_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question only from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\nContext:\n{context}")
])

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

vectorstore = setup_pipeline(file_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})

chain = parallel | prompt | model | StrOutputParser()
q = "what is machine learning and tell me its types"
config = {
    "run_name": "pdf_rag_query"
}
ans = chain.invoke(q)
print("Answer : ", ans)
