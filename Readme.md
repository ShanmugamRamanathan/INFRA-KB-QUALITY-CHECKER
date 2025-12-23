```markdown
# Infrastructure Knowledge Base Quality Checker

A production-grade RAG (Retrieval-Augmented Generation) system that evaluates knowledge base quality for IT infrastructure troubleshooting. Built with Qdrant, Ollama, and Streamlit.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Problem

IT teams maintain scattered documentation across wikis, OneNote, Confluence, and text files. Questions remain:
- Does our KB actually answer common support tickets?
- Which topics need better documentation?
- How do we measure KB quality objectively?

This tool answers those questions with quantitative metrics.

## Solution

An end-to-end system that:
1. Retrieves relevant KB articles using semantic search
2. Generates answers via local LLM
3. Evaluates quality with 4 key metrics
4. Provides an interactive dashboard for real-time Q&A

## Architecture

```
User Question
    ↓
BGE Embeddings (384-dim)
    ↓
Qdrant Vector Search (top-K docs)
    ↓
Ollama LLM (llama3.2) → Answer
    ↓
Evaluation Layer → Metrics
    ↓
Streamlit Dashboard
```

**Tech Stack:**
- **Vector DB:** Qdrant (local)
- **Embeddings:** BAAI/bge-small-en-v1.5
- **LLM:** Ollama (llama3.2)
- **Frontend:** Streamlit
- **Language:** Python 3.10+

## Features

- 🔍 Semantic search over infrastructure KB
- 🤖 Local LLM-based Q&A (no API costs)
- 📊 Real-time quality metrics per question
- 📚 Source transparency (shows retrieved docs)
- 💾 Session export for analysis
- ⚙️ Configurable retrieval settings

## Evaluation Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Context Relevancy** | % of retrieved content actually relevant | >0.7 |
| **Answer Completeness** | How fully the question is answered | >0.8 |
| **Faithfulness** | Answer grounded in sources (no hallucination) | >0.9 |
| **Precision@K** | Quality of top-K retrieved documents | >0.6 |
| **Overall Score** | Weighted combination of above | >0.7 |

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Qdrant** (via Docker):
   ```
   docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
   ```

3. **Ollama**:
   ```
   # Install from https://ollama.com
   ollama pull llama3.2
   ```

### Setup

```
git clone https://github.com/yourusername/infra-kb-quality-checker.git
cd infra-kb-quality-checker

pip install -r requirements.txt
```

## Quick Start

### 1. Load Knowledge Base

```
python scripts/load_kb_into_qdrant.py
```

**Expected output:**
```
Created collection: infra_kb
Loaded 10 snippets from KB
Model moved to GPU
Inserted 10 points into infra_kb

Search results for: SCOM agent heartbeat failure
Score: 0.839
Text: [DOC1] Title: SCOM Heartbeat Failure...
```

### 2. Launch Dashboard

```
streamlit run app/streamlit_dashboard.py
```

Opens at `http://localhost:8501`

### 3. Ask Questions

Try these sample questions:
- `Why is my SCOM agent not sending heartbeats?`
- `How do I troubleshoot high CPU usage on a monitored server?`
- `What should I do if SCOM database backup fails?`

## Project Structure

```
infra-kb-quality-checker/
├── app/
│   ├── streamlit_dashboard.py      # Interactive Q&A dashboard
│   └── evaluation_metrics.py       # Metric calculations
├── data/
│   └── kb_docs/
│       └── infra_sample_kb.txt     # Sample KB (SCOM/SCCM/Windows)
├── scripts/
│   ├── load_kb_into_qdrant.py      # KB vectorization & ingestion
│   ├── check_kb_quality.py         # Batch evaluation CLI
│   └── evaluation_metrics.py       # Shared metrics module
├── reports/                         # Generated evaluation reports
├── requirements.txt
└── README.md
```

## How It Works

### 1. Knowledge Base Loading
- Reads KB text file (`infra_sample_kb.txt`)
- Splits into document snippets
- Generates 384-dim embeddings using BGE model
- Stores in Qdrant vector database

### 2. Question Processing
```
Question → Embed → Search Qdrant → Retrieve top-3 docs
```

### 3. Answer Generation
```
Retrieved docs + Question → Ollama (llama3.2) → Answer
```

### 4. Quality Evaluation
LLM analyzes:
- Are retrieved docs relevant?
- Does KB fully answer the question?
- Is the answer faithful to sources?
- Were the right documents retrieved?

## Sample Output

```
🙋 You asked: Why is my SCOM agent not sending heartbeats?

🤖 Answer:
According to Source 1, when a SCOM agent stops sending heartbeats, first check 
if the server is online and reachable. Verify DNS resolution works correctly. 
Then confirm the Microsoft Monitoring Agent (Health Service) is running and 
restart it if needed using services.msc or PowerShell...

📊 Quality Metrics:
Overall: 0.43  Relevancy: 0.13🔴  Complete: 0.80🟢  Faithful: 0.60🟡  Precision: 0.00🔴

📚 3 Source Documents Used
```

## Batch Evaluation (CLI)

For automated testing:

```
python scripts/check_kb_quality.py
```

Generates:
- `reports/kb_quality_detailed_YYYYMMDD_HHMMSS.json`
- `reports/kb_quality_summary_YYYYMMDD_HHMMSS.json`

Example summary output:
```
{
  "total_questions": 3,
  "avg_metrics": {
    "context_relevancy": 0.258,
    "answer_completeness": 0.800,
    "faithfulness": 0.600,
    "precision_at_3": 0.000,
    "overall_score": 0.464
  },
  "worst_questions": [...]
}
```

## Configuration

Edit these constants in scripts:

```
# Qdrant connection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "infra_kb"

# Models
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"
```

## Extending the Project

Ideas for enhancement:
- [ ] Add 50+ KB articles to improve metrics
- [ ] Support multiple KB collections (Linux, Network, Cloud)
- [ ] Fine-tune embeddings on domain data
- [ ] Add user authentication
- [ ] Deploy with Docker Compose
- [ ] Integrate with Slack/Teams for team access
- [ ] A/B test different LLMs (GPT-4, Claude, Mistral)

## Why This Project Matters

Demonstrates:
- ✅ Production RAG architecture (not just a chatbot)
- ✅ Quantitative evaluation (not subjective judgment)
- ✅ Real business value (identifies KB gaps)
- ✅ Full-stack ML (embeddings + vector DB + LLM + UI)
- ✅ Local deployment (no API costs, data stays private)

Perfect for portfolios targeting: MLOps, AI Engineering, SRE/DevOps automation, Enterprise AI roles.

## Results

Current baseline (10 KB documents):

| Metric | Score | Status |
|--------|-------|--------|
| Context Relevancy | 0.258 | 🔴 Needs improvement |
| Answer Completeness | 0.800 | 🟢 Good |
| Faithfulness | 0.600 | 🟡 Acceptable |
| Precision@3 | 0.000 | 🔴 Critical - need more docs |
| **Overall** | **0.464** | 🟠 **Fair** |

**Key insight:** System correctly identifies that KB needs 2-3x more content to adequately answer common questions.


## Author

[Shanmugam Ramanathan](https://github.com/ShanmugamRamanathan)  li
[LinkedIn](https://www.linkedin.com/in/shanmugam-ramanathan-260953262/)

[1](https://towardsdatascience.com/dont-build-an-ml-portfolio-without-these-projects/)
[2](https://www.guvi.in/blog/machine-learning-professional-portfolio/)
