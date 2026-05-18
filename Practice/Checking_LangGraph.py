import inspect
from langgraph.graph import StateGraph

print([m for m in dir(StateGraph) if 'conditional' in m.lower()])
print(inspect.signature(StateGraph.add_conditional_edges))