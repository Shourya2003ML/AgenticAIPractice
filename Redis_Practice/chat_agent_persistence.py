# A simple implementation of redis short term memory persistence for agent
from typing import Literal
import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.redis import RedisSaver

from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information"""
    if city == "nyc":
        return "It might be cloudy in NYC"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown City")
    
#Setting up tools and model
tools = [get_weather]
model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0)

#Creating redis persistence
REDIS_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
    #Initialize Redis indices
    checkpointer.setup()

    #Create agent with memory
    graph = create_agent(model, tools = tools, checkpointer = checkpointer)

    #Use the agent to create thread ID to maintain conversation state
    config = {"configurable": {"thread_id": "user123"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    print(res["messages"][-1].content)