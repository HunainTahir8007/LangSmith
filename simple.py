from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model =  ChatGroq(model = "llama-3.3-70b-versatile" , temperature= 0.2)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
