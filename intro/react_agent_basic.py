from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
from datetime import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
# result = llm.invoke("whats the weather in nairobi")

# print(result)

search_tool= TavilySearchResults(search_depth="basic")


@tool
def get_current_time_and_date(query: str) -> str:
    """Returns the current time and date."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")




tools = [search_tool, get_current_time_and_date]

agent = initialize_agent(tools = tools, llm=llm, agent ="zero-shot-react-description", verbose=True )

agent.invoke("Whats the time right now")