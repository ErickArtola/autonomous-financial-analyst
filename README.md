# Autonomous Financial Research Analyst

An end-to-end agentic AI system that autonomously researches and ranks investment opportunities in the AI sector. Built with **LangGraph**, **LangChain**, and **RAG (Retrieval-Augmented Generation)**.

---

## Overview

Traditional LLMs answer questions reactively. This project builds a **goal-oriented autonomous agent** that proactively gathers data from multiple sources, synthesises insights from private knowledge bases, and produces structured investment reports — without step-by-step human instruction.

The agent is implemented in two parts:

### Part 1 — Agent Architecture & Tool Orchestration
- **Proactive agent charter** that defines mission, constraints, and quality standards
- **4 specialised tools** (actuators) wired into a LangGraph state machine:
  - `get_stock_price` — real-time price, volume, and market cap via Yahoo Finance
  - `get_stock_history` — 3-year performance trends and return calculations
  - `search_financial_news` — live web search via Tavily
  - `analyze_sentiment` — GPT-powered sentiment classification on news text
- **Reactive error handling** — agent adapts when a tool fails (demonstrated with a simulated failure test)
- **Persistent memory** via LangGraph's `MemorySaver` (thread-scoped conversation state)

### Part 2 — RAG Pipeline & Multi-Company Ranking
- **RAG pipeline** over a corpus of private company AI-initiative PDFs:
  - `PyPDFDirectoryLoader` → `RecursiveCharacterTextSplitter` → OpenAI embeddings → ChromaDB
  - `query_private_database` tool enables semantic retrieval at agent decision time
- **Synergistic tool usage** — agent chains news search → sentiment analysis → RAG query in a single reasoning loop
- **Investment ranking system** — multi-company comparative analysis across financial performance and AI research activity (MSFT, GOOGL, NVDA, AMZN, IBM)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│           LangGraph Agent           │
│                                     │
│  ┌──────────┐    ┌───────────────┐  │
│  │  Agent   │◄──►│  Tool Node   │  │
│  │  Node    │    │              │  │
│  │ (GPT-4o) │    │ ┌──────────┐ │  │
│  └──────────┘    │ │ Stock    │ │  │
│        │         │ │ Price    │ │  │
│   conditional    │ ├──────────┤ │  │
│   edge           │ │ History  │ │  │
│        │         │ ├──────────┤ │  │
│        ▼         │ │ News     │ │  │
│      [END]       │ ├──────────┤ │  │
│                  │ │Sentiment │ │  │
│                  │ ├──────────┤ │  │
│                  │ │ RAG DB   │ │  │
│                  │ └──────────┘ │  │
│                  └───────────────┘  │
└─────────────────────────────────────┘
         │                │
    Yahoo Finance    ChromaDB + PDFs
    Tavily Search    OpenAI Embeddings
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 0.3 |
| LLM | GPT-4o-mini via OpenAI API |
| Tool orchestration | LangChain 0.3 |
| Web search | Tavily |
| Financial data | yfinance |
| Vector store | ChromaDB |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Document loading | LangChain `PyPDFDirectoryLoader` |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd autonomous-financial-analyst
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp env.example .env
# Edit .env and add your OPENAI_API_KEY and TAVILY_API_KEY
```

Get your keys:
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://app.tavily.com/

### 3. Run the notebook

```bash
jupyter notebook Autonomous_financial_analyst_Learners_Notebook.ipynb
```

Run cells top-to-bottom. The notebook is self-contained — all sections build on each other.

---

## Key Concepts Demonstrated

- **Agentic AI design patterns** — proactiveness, autonomy, reactivity, goal-orientation
- **LangGraph state machines** — nodes, conditional edges, memory checkpointing
- **RAG implementation** — chunking strategy, embedding generation, semantic retrieval
- **Tool composition** — combining real-time data, web search, and private knowledge in one agent loop
- **Error resilience** — graceful degradation when tools fail, with continued task completion
