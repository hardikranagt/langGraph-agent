from dotenv import load_dotenv
load_dotenv()  # must be before ChatOpenAI is initialized

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────
# 1. State Definition
# ─────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messages = append, not overwrite

# ─────────────────────────────────────────
# 2. LLM Setup
# ─────────────────────────────────────────
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOllama(model="llama3", temperature=0)

# ─────────────────────────────────────────
# 3. Node — calls the LLM
# ─────────────────────────────────────────
def call_llm(state: State) -> dict:
    """Node: sends messages to LLM, returns response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ─────────────────────────────────────────
# 4. Build the Graph
# ─────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(State)

    # Register nodes
    graph.add_node("llm", call_llm)

    # Entry point and edges
    graph.set_entry_point("llm")
    graph.add_edge("llm", END)

    return graph.compile()

# ─────────────────────────────────────────
# 5. Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    print("LangGraph Simple Agent — type 'exit' to quit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        response = result["messages"][-1].content
        print(f"Agent: {response}\n")
