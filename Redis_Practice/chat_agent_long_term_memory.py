import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.redis import RedisStore
from langgraph.store.base import BaseStore

from dotenv import load_dotenv

load_dotenv()

#Setting up the model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature = 0)

#function to store and access and save user memories
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)

    #Retrieving relevant memories from the user
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assisstant talking to the user. User info: {info}"

    #Store new memories if the user asks to remember something
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke([{"role": "system", "content": system_msg}] + state["messages"])
    return {"messages": response}

#Build the graph
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

#Initialize Redis Persistence and store
REDIS_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(REDIS_URI) as checkpointer:
    checkpointer.setup()

    with RedisStore.from_conn_string(REDIS_URI) as store:
        store.setup()

        #Compile graph with checkpointer and store
        graph = builder.compile(checkpointer = checkpointer, store = store)

        # First conversation - tell the agent to remember something
        config = {"configurable": {"thread_id": "convo1", "user_id": "user123"}}
        response = graph.invoke(
            {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
            config
        )

        # Second conversation - different thread but same user
        new_config = {"configurable": {"thread_id": "convo2", "user_id": "user123"}}
        response = graph.invoke(
            {"messages": [{"role": "user", "content": "What's my name?"}]},
            new_config
        )
        # Agent will respond with "Your name is Bob"
        print(response["messages"][-1].content)
