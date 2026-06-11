# ============================================================
# LangGraph Agent — Tools + Conditional Edges + Memory
# ============================================================
# Covers everything learned so far in one runnable file:
#   1. State with custom fields
#   2. Tools with @tool decorator
#   3. LLM node with bound tools
#   4. Conditional edges (router)
#   5. ToolNode (prebuilt)
#   6. MemorySaver (short-term memory across turns)
#   7. Summarization node (keep token usage low)
# ============================================================

# Install dependencies:
# pip install langgraph langchain-openai langchain-core

import os
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0)

# ─────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────
# Everything that flows through the graph lives here.
# add_messages = appends new messages instead of overwriting.

class State(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    user_name: str                           # remembered across turns
    summary: str                             # compressed older messages


# ─────────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Mock — replace with a real weather API call
    weather_data = {
        "mumbai": "32°C, humid and partly cloudy",
        "delhi": "38°C, hot and sunny",
        "bangalore": "24°C, pleasant with light breeze",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression like '2 + 2' or '100 * 0.18'."""
    try:
        # Only allow safe math characters
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Only basic math operators allowed."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def remember_name(name: str) -> str:
    """Store the user's name when they introduce themselves."""
    return f"Got it! I'll remember your name is {name}."


tools = [get_weather, calculator, remember_name]


# ─────────────────────────────────────────────
# 3. LLM — bound to tools
# ─────────────────────────────────────────────

llm = ChatOllama(model="llama3.1", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────
# 4. NODES
# ─────────────────────────────────────────────

def llm_node(state: State) -> dict:
    """
    Core LLM node.
    - Injects summary as context if available
    - Injects user name into system prompt if known
    """
    system_parts = ["You are a helpful assistant."]

    if state.get("user_name"):
        system_parts.append(f"The user's name is {state['user_name']}.")

    if state.get("summary"):
        system_parts.append(f"Summary of earlier conversation: {state['summary']}")

    system_msg = SystemMessage(content=" ".join(system_parts))
    response = llm_with_tools.invoke([system_msg, *state["messages"]])

    return {"messages": [response]}


def tool_node_fn(state: State) -> dict:
    """
    Wraps LangGraph's prebuilt ToolNode.
    Also extracts user_name from remember_name tool result.
    """
    # Run the tool
    result = ToolNode(tools).invoke(state)

    # Check if remember_name was called — extract name from tool result message
    updated_name = state.get("user_name", "")
    for msg in result.get("messages", []):
        if hasattr(msg, "name") and msg.name == "remember_name":
            # Tool returned "Got it! I'll remember your name is Raj."
            content = msg.content
            if "is " in content:
                updated_name = content.split("is ")[-1].rstrip(".")

    return {**result, "user_name": updated_name}


def summarize_node(state: State) -> dict:
    """
    Summarizes old messages when conversation gets long (>10 messages).
    Keeps only last 4 messages + stores summary in state.
    """
    summary_prompt = f"""
    Summarize the conversation below in 3-4 sentences.
    Preserve key facts: names, decisions, questions asked, answers given.

    Conversation:
    {[m.content for m in state["messages"]]}
    """

    summary = llm.invoke(summary_prompt).content
    recent_messages = state["messages"][-4:]  # keep last 4 only

    print(f"\n[Memory] Summarized {len(state['messages'])} messages → keeping last 4.\n")

    return {
        "messages": recent_messages,
        "summary": summary,
    }


# ─────────────────────────────────────────────
# 5. CONDITIONAL EDGES (routers)
# ─────────────────────────────────────────────

def should_use_tool(state: State) -> str:
    """After LLM responds — did it request a tool call?"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "check_memory"   # go check if we need to summarize


def should_summarize(state: State) -> str:
    """After tool or LLM response — is conversation getting too long?"""
    if len(state["messages"]) > 10:
        return "summarize"
    return END


# ─────────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────────

graph = StateGraph(State)

# Register nodes
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", tool_node_fn)
graph.add_node("check_memory", lambda state: state)   # passthrough node
graph.add_node("summarize_node", summarize_node)

# Entry point
graph.set_entry_point("llm_node")

# After LLM → use tool OR check memory
graph.add_conditional_edges(
    "llm_node",
    should_use_tool,
    {
        "tool_node": "tool_node",
        "check_memory": "check_memory",
    }
)

# After tool → always go back to LLM
graph.add_edge("tool_node", "llm_node")

# After check_memory → summarize if needed, else END
graph.add_conditional_edges(
    "check_memory",
    should_summarize,
    {
        "summarize": "summarize_node",
        END: END,
    }
)

# After summarize → END (next turn starts fresh from llm_node)
graph.add_edge("summarize_node", END)

# Compile with MemorySaver — enables multi-turn memory per thread_id
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# ─────────────────────────────────────────────
# 7. RUN — multi-turn conversation
# ─────────────────────────────────────────────

def chat(thread_id: str, user_input: str) -> str:
    """Send a message and get a reply, maintaining memory per thread."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    thread = "session_hardik_001"

    print("Turn 1:")
    print(chat(thread, "Hi! My name is Hardik."))

    print("\nTurn 2:")
    print(chat(thread, "What's the weather in Ahmedabad?"))

    print("\nTurn 3:")
    print(chat(thread, "What is 18% of 4500?"))

    print("\nTurn 4 — tests memory:")
    print(chat(thread, "Do you remember my name?"))


# ─────────────────────────────────────────────
# EXPECTED OUTPUT
# ─────────────────────────────────────────────
#
# Turn 1:
# Nice to meet you, Raj! How can I help you today?
#
# Turn 2:
# The weather in Mumbai is 32°C, humid and partly cloudy.
#
# Turn 3:
# 18% of 4500 is 810.
#
# Turn 4:
# Yes! Your name is Raj.
#
# ─────────────────────────────────────────────
# GRAPH FLOW DIAGRAM
# ─────────────────────────────────────────────
#
#  [START]
#     │
#  [llm_node]  ◄─────────────────────────┐
#     │                                   │
#  tool called?                           │
#  ├── YES → [tool_node] ─────────────────┘
#  └── NO  → [check_memory]
#                 │
#           >10 messages?
#           ├── YES → [summarize_node] → END
#           └── NO  → END
#
# ─────────────────────────────────────────────