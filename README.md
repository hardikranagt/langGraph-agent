# LangGraph Starter — Simple ReAct Agent

A minimal LangGraph agent to understand the core building blocks:
**State → Node → Graph → Run**

---

## Installation Steps

### 1. Prerequisites
- Python 3.10 or higher
- An OpenAI API key

### 2. Clone / Download the project
```bash
cd langgraph_starter
```

### 3. Create a virtual environment
```bash
python -m venv venv

# Activate it:
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up your API key
```bash
cp .env.example .env
# Open .env and replace with your actual OpenAI API key
```

### 6. Run the agent
```bash
python src/agent.py
```

---

## Project Structure

```
langgraph_starter/
├── src/
│   └── agent.py        # Main agent (State, Node, Graph)
├── .env.example        # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How It Works

```
[START] → [llm node] → [END]
```

| Concept   | What it does                              |
|-----------|-------------------------------------------|
| `State`   | Shared dict with `messages` list          |
| `Node`    | `call_llm()` — invokes OpenAI, returns response |
| `Graph`   | Wires nodes together with edges           |
| `compile()` | Locks the graph, returns a runnable app |

---

## What's Next?

This is Step 1 of your LangGraph learning path:

1. ✅ **State & Nodes** ← you are here  
2. ✅ Conditional Edges — route based on LLM output  
3. ✅ Tool Nodes — connect tools (like CrewAI tools)  
4. ✅ Cycles & Human-in-the-loop  
5. ✅ Multi-agent graphs  
6. ✅ LangGraph + RAG  
