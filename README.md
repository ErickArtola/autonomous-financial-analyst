# Autonomous Financial Research Analyst

An intelligent, goal-oriented agent that autonomously researches and ranks AI-sector investment opportunities using **LangGraph**, **LangChain**, **RAG**, and **OpenAI's GPT-4o-mini**.

---

## 🎯 Project Overview

This project demonstrates an end-to-end **agentic AI system** that goes beyond simple question-answering. Given a company ticker, the agent independently decides which tools to call, in what order, and how to synthesise results into a structured investment report — without step-by-step instructions.

**Key Companies Analysed:** Google (GOOGL), Microsoft (MSFT), IBM, NVIDIA (NVDA), Amazon (AMZN)

### Problem Statement

Traditional LLMs suffer from critical limitations in financial analysis:
- **Hallucinations**: Plausible but factually incorrect outputs
- **Knowledge Cutoff**: Limited to training data timestamp — no real-time prices or news
- **No Source Verification**: Cannot cite or validate information
- **Reactivity**: Waits to be told exactly what to do instead of taking initiative
- **No Private Knowledge**: Cannot access internal analyst reports or proprietary documents

### Solution: Autonomous Agent + RAG

The system combines two architectures to address these limitations:

1. **LangGraph Agent** — A goal-oriented state machine that proactively gathers all required data before reporting, adapts when tools fail, and maintains conversation memory across turns
2. **RAG Pipeline** — Indexes private company AI-initiative PDFs into ChromaDB, enabling semantic retrieval at agent decision time alongside live market data

This enables:
- ✅ Real-time stock data and news (no knowledge cutoff)
- ✅ Grounded responses with source citations from private documents
- ✅ Proactive, multi-step reasoning without hand-holding
- ✅ Resilient error handling when individual tools fail

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           LangGraph Agent               │
│                                         │
│  ┌──────────────┐   ┌────────────────┐  │
│  │  Agent Node  │◄─►│   Tool Node    │  │
│  │  (GPT-4o-    │   │                │  │
│  │   mini)      │   │ ┌────────────┐ │  │
│  └──────────────┘   │ │Stock Price │ │  │
│         │           │ ├────────────┤ │  │
│   conditional       │ │3Y History  │ │  │
│   routing           │ ├────────────┤ │  │
│         │           │ │News Search │ │  │
│         ▼           │ ├────────────┤ │  │
│       [END]         │ │Sentiment   │ │  │
│                     │ ├────────────┤ │  │
│                     │ │RAG Query   │ │  │
│                     │ └────────────┘ │  │
│                     └────────────────┘  │
└─────────────────────────────────────────┘
         │                    │
    Yahoo Finance         ChromaDB
    Tavily Search      OpenAI Embeddings
```

**Two-Part Implementation:**

**Part 1 — Agent Architecture & Tool Orchestration**
- Goal-oriented agent charter defining mission, constraints, and quality standards
- 4 specialised tools wired into a LangGraph state machine
- Reactive error handling demonstrated with a simulated tool-failure test
- Persistent memory via LangGraph's `MemorySaver` (thread-scoped)

**Part 2 — RAG Pipeline & Multi-Company Ranking**
- Full RAG pipeline: PDF loading → chunking → OpenAI embeddings → ChromaDB → semantic retrieval
- Synergistic tool usage: news search → sentiment analysis → RAG query in a single agent loop
- Investment ranking system across 5 companies on financial + AI research dimensions

---

## 🛠️ Technology Stack

### AI / Agent Framework
- **LangGraph 0.3**: State machine orchestration, conditional routing, memory checkpointing
- **LangChain 0.3**: Tool definitions, prompt management, LLM chain abstractions
- **OpenAI GPT-4o-mini**: Cost-effective LLM for reasoning and generation

### Data & Search
- **yfinance 0.2.66**: Real-time stock prices, historical data, market metrics
- **Tavily Search**: Live web search for financial news

### RAG Stack
- **ChromaDB 1.3.4**: Vector database for semantic search over private documents
- **OpenAI `text-embedding-ada-002`**: Document and query embeddings
- **LangChain PyPDFDirectoryLoader**: Automated PDF ingestion
- **RecursiveCharacterTextSplitter**: Tiktoken-aware document chunking

### Configuration
- **python-dotenv**: Environment variable management (API keys never hardcoded)

---

## 📋 Project Structure

```
autonomous-financial-analyst/
├── Autonomous_financial_analyst_Learners_Notebook.ipynb  # Main notebook
├── Companies-AI-Initiatives/                             # RAG knowledge base
│   ├── AMZN.pdf
│   ├── GOOGL.pdf
│   ├── IBM.pdf
│   ├── MSFT.pdf
│   └── NVDA.pdf
├── requirements.txt                                      # Python dependencies
├── env.example                                           # API key template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- OpenAI API key (GPT-4o-mini access)
- Tavily API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ErickArtola/autonomous-financial-analyst.git
   cd autonomous-financial-analyst
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Copy the environment template**
   ```bash
   cp env.example .env
   ```

2. **Add your API keys to `.env`**
   ```
   OPENAI_API_KEY=sk-your-key-here
   TAVILY_API_KEY=tvly-your-key-here
   ```

   Get your keys:
   - OpenAI: https://platform.openai.com/api-keys
   - Tavily: https://app.tavily.com/

   > ⚠️ Never commit `.env` to GitHub — it is already covered by `.gitignore`

### Run the Notebook

```bash
jupyter notebook Autonomous_financial_analyst_Learners_Notebook.ipynb
```

Run cells top-to-bottom. Each section builds on the previous.

### Expected Output

```
TEST 3: Full Autonomous Agent (With All Constraints)
================================================================================

Query: Provide a comprehensive investment analysis for Microsoft (MSFT)

🤖 FULL AGENT RESPONSE:
================================================================================
## Investment Research Briefing: Microsoft Corporation (MSFT)

### 1. Market Position
- Current Price: $415.32 USD
- Market Cap: $3.08T
- Volume: 22,847,391

### 2. Performance (3-Year)
- 3-Year Return: +64.2%
- Start Price: $252.56 → Current: $415.32

### 3. News Sentiment
- Sentiment: Positive (confidence: 0.91)
- Key themes: Azure AI growth, Copilot adoption, enterprise cloud expansion

### 4. AI Research Activity
- Azure AI Foundry Labs, OpenAI partnership, GitHub Copilot
- Deep integration across enterprise product suite

### 5. Investment Recommendation
- Rating: BUY | Confidence: High
- Key risk: Premium valuation relative to peers
```

---

## 🔑 Key Components

### 1. Agent State & Graph

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
agent = workflow.compile(checkpointer=MemorySaver())
```

### 2. Financial Data Tools

```python
@tool
def get_stock_price(ticker: str) -> Dict:
    """Real-time price, market cap, and volume via Yahoo Finance."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {"current_price": info.get("currentPrice"), "market_cap": info.get("marketCap"), ...}

@tool
def get_stock_history(ticker: str, period: str = "3y") -> Dict:
    """3-year return, trend direction, and data points."""
    hist = yf.Ticker(ticker).history(period=period)
    total_return = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
    return {"total_return_pct": round(total_return, 2), ...}
```

### 3. RAG Pipeline

```python
# Index private documents
loader = PyPDFDirectoryLoader("Companies-AI-Initiatives/")
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(loader.load())
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(), collection_name="AI_Initiatives")
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# RAG tool used by the agent
@tool
def query_private_database(query: str) -> str:
    """Semantic search over internal AI-initiative documents."""
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    return llm.invoke(f"Based on these documents, answer: {query}\n\nContext:\n{context}").content
```

### 4. Multi-Company Ranking

```python
companies = ["MSFT", "GOOGL", "NVDA", "AMZN", "IBM"]
query = f"Rank {companies} from most to least valuable for investment based on 3-year performance and AI research activity."
response = agent.invoke({"messages": [HumanMessage(content=query)]}, config)
```

---

## 📈 Use Cases

The agentic RAG architecture demonstrated here applies broadly across industries:

| Domain | Application |
|---|---|
| **Financial Advisory** | Personalised recommendations with real-time market data and private research |
| **Customer Support** | Grounded responses with source citations from internal knowledge bases |
| **Medical Research** | Clinical trial data retrieval combined with patient-specific EHR context |
| **Legal Analysis** | Contract databases, regulatory compliance checking, case law retrieval |
| **Enterprise Intelligence** | Synthesis of internal reports with live external data sources |

---

## 🧪 Evaluation Dimensions

| Metric | Definition |
|---|---|
| **Faithfulness** | Does the response accurately reflect retrieved documents? |
| **Answer Relevance** | Does the recommendation directly address the query? |
| **Context Relevance** | Are retrieved chunks relevant to the ticker queried? |
| **Tool Coverage** | Did the agent invoke all required tools before reporting? |

For production systems, these are measured using an **LLM-as-a-Judge** pattern or a framework like [RAGAS](https://github.com/explodinggradients/ragas).

---

## ⚙️ Configuration Options

### Model Selection

```python
# Default (cost-optimised)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Higher capability
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

### RAG Parameters

```python
# Adjust in the chunking cell
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,    # Tokens per chunk
    chunk_overlap=200   # Overlap between chunks
)

# Adjust retrieval breadth
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Top-k results
```

---

## 🔧 Troubleshooting

**`Missing required environment variables`**
Ensure `.env` exists in the project root with both `OPENAI_API_KEY` and `TAVILY_API_KEY` set.

**`No module named 'langchain_tavily'`**
```bash
pip install -U langchain-tavily
```

**PDF documents not found**
Confirm the `Companies-AI-Initiatives/` folder is present in the project root with all 5 PDFs.

**ChromaDB errors after reinstall**
```bash
pip uninstall chromadb -y && pip install chromadb==1.3.4
```

**Rate limit errors from OpenAI**
The agent makes multiple LLM calls per run. If you hit rate limits, add a short `time.sleep(2)` between test cells or upgrade your OpenAI tier.

---

## 💰 Cost Considerations

| Model | Input | Output |
|---|---|---|
| gpt-4o-mini | $0.15 / 1M tokens | $0.60 / 1M tokens |
| gpt-4o | $5.00 / 1M tokens | $15.00 / 1M tokens |

**Estimated cost for this project:**
- Single company analysis: ~$0.02–0.05
- Full 5-company ranking run: ~$0.10–0.20
- 100 analyses/day: ~$2–5/day

**Cost optimisation tips:**
- Use `gpt-4o-mini` (default) — 30× cheaper than `gpt-4o` for this task
- Cache vector embeddings (generated once, reused across runs)
- Monitor token usage via `response.usage_metadata`

---

## 🔐 Security Best Practices

✅ Store API keys in `.env` (git-ignored)  
✅ Use `env.example` to document required variables without exposing values  
✅ Rotate API keys regularly  
✅ Validate inputs before passing to LLM  

❌ Never commit `.env` or `config.json` to GitHub  
❌ Never hardcode API keys in notebook cells  
❌ Never log API keys or raw credentials  

---

## 🤝 Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-improvement`
3. Commit your changes: `git commit -m "Add your improvement"`
4. Push to the branch: `git push origin feature/your-improvement`
5. Open a Pull Request

**Areas for contribution:**
- [ ] Additional financial metrics (P/E ratio, dividend yield, beta)
- [ ] Streaming agent responses via FastAPI + LangServe
- [ ] Web dashboard for analysis visualisation
- [ ] Expanded company coverage (S&P 500)
- [ ] LangSmith evaluation harness for RAG quality metrics
- [ ] Persistent ChromaDB storage (currently in-memory)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Erick Artola**  
AI/ML Engineer | Agentic AI

- **LinkedIn**: [linkedin.com/in/erick-artola](https://www.linkedin.com/in/erick-artola)
- **GitHub**: [github.com/ErickArtola](https://github.com/ErickArtola)

---

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and the embeddings API
- LangChain & LangGraph teams for the agent orchestration framework
- ChromaDB for the vector database
- yfinance for financial data access
- Tavily for real-time search

---

**Last Updated**: April 2026  
**Status**: Active  
**Maintained by**: Erick Artola
