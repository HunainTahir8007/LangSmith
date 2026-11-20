from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI


load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", gt=0, lt=10)

structured_model = model.with_structured_output(EvaluationSchema)