import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

os.environ["LANGSMITH_PROJECT"] = "" # project name 
load_dotenv()  

PDF_PATH = ""   # your file path

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()  


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
splits = splitter.split_documents(docs)

emb = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

model =  ChatGroq(model = "llama-3.3-70b-versatile" , temperature= 0.2)
def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | model | StrOutputParser()

q = "What is machine learning"
ans = chain.invoke(q.strip())
print("ANS", ans)
