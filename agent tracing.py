from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=f07d9636974c4120025fadf60678771b&query={city}'

  response = requests.get(url)

  return response.json()

llm = ChatOpenAI()

prompt = hub.pull("hwchase17/react")   # fetch the react agent prompt for the langchain hub

agent = create_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

response = agent.invoke({"input": "What is the current temp of Lahore"})

print(response['output'])