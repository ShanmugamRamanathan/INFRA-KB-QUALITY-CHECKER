```markdown
# Infrastructure Knowledge Base Quality Checker

A production-grade RAG (Retrieval-Augmented Generation) system that evaluates knowledge base quality for IT infrastructure troubleshooting. Built with Qdrant, Ollama, and Streamlit.

---

## Problem

IT teams maintain scattered documentation across wikis, OneNote, Confluence, and text files. Common questions arise:

- Does our knowledge base actually answer real support tickets?
- Which topics need better documentation?
- How can we measure KB quality objectively?

This project answers those questions using **quantitative evaluation metrics**.

---

## Solution

An end-to-end system that:

1. Retrieves relevant KB articles using semantic search  
2. Generates answers using a local LLM  
3. Evaluates answer quality using multiple metrics  
4. Provides an interactive dashboard for real-time analysis  

---

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

### Tech Stack

- **Vector DB:** Qdrant (local)
- **Embeddings:** BAAI/bge-small-en-v1.5
- **LLM:** Ollama (llama3.2)
- **Frontend:** Streamlit
- **Language:** Python 3.10+

---

## Features

- 🔍 Semantic search over infrastructure KB
- 🤖 Local LLM-based Q&A (no API costs)
- 📊 Real-time quality metrics per question
- 📚 Source transparency (retrieved documents shown)
- 💾 Session export for offline analysis
- ⚙️ Configurable retrieval parameters

---

## Evaluation Metrics

| Metric | Description | Target |
|------|------------|--------|
| **Context Relevancy** | Relevance of retrieved content | > 0.7 |
| **Answer Completeness** | How fully the question is answered | > 0.8 |
| **Faithfulness** | Answer grounded in sources | > 0.9 |
| **Precision@K** | Quality of top-K retrieval | > 0.6 |
| **Overall Score** | Weighted aggregate score | > 0.7 |

---

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Qdrant (Docker)**

```

docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant

```

3. **Ollama**

```

# Install from [https://ollama.com](https://ollama.com)

ollama pull llama3.2

```

---

### Setup

```

git clone [https://github.com/yourusername/infra-kb-quality-checker.git](https://github.com/yourusername/infra-kb-quality-checker.git)
cd infra-kb-quality-checker
pip install -r requirements.txt

```

---

## Quick Start

### 1. Load Knowledge Base

```

python scripts/load_kb_into_qdrant.py

```

### 2. Launch Dashboard

```

streamlit run app/streamlit_dashboard.py

```

Open: `http://localhost:8501`

### 3. Ask Questions

Sample questions:

- Why is my SCOM agent not sending heartbeats?
- How do I troubleshoot high CPU usage?
- What causes SCOM database backup failures?

---

## Project Structure

```

infra-kb-quality-checker/
├── app/
│   ├── streamlit_dashboard.py
│ 
├── data/
│   └── kb_docs/
│       └── infra_sample_kb.txt
├── scripts/
│   ├── load_kb_into_qdrant.py
│   ├── check_kb_quality.py
│   └── evaluation_metrics.py
├── reports/
├── requirements.txt
└── README.md

```

---

## How It Works

### Knowledge Base Ingestion
- KB text is chunked into snippets
- Embeddings generated using BGE
- Stored in Qdrant vector database

### Question Flow
```

Question → Embed → Vector Search → Top-K Docs

```

### Answer Generation
```

Docs + Question → Ollama → Final Answer

```

### Evaluation
- Context relevancy
- Answer completeness
- Faithfulness to sources
- Retrieval precision

---

## Sample Output

```

🙋 Question:
Why is my SCOM agent not sending heartbeats?

🤖 Answer:
Check if the server is reachable, verify DNS resolution,
ensure the Microsoft Monitoring Agent service is running,
and restart the Health Service if needed.

📊 Metrics:
Overall: 0.46
Relevancy: 0.13
Completeness: 0.80
Faithfulness: 0.60
Precision@3: 0.00

```

---

## Batch Evaluation (CLI)

```

python scripts/check_kb_quality.py

```

Generates:

- Detailed JSON report
- Summary metrics report
- Worst-performing questions

---

## Configuration

```

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "infra_kb"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"

```

---

## Future Enhancements

- Add larger KB datasets
- Multiple KB collections (Linux, Network, Cloud)
- Docker Compose deployment
- Authentication & role-based access
- Slack / Teams integration
- LLM comparison (GPT, Mistral, Claude)

---

## Why This Project Matters

- ✅ Real-world RAG system
- ✅ Objective KB quality evaluation
- ✅ Enterprise-focused use case
- ✅ End-to-end AI pipeline
- ✅ Local, private, cost-free inference

Ideal for portfolios targeting **AI Engineer, MLOps, SRE, DevOps, and Enterprise AI roles**.

---

## Author

**Shanmugam Ramanathan**  
GitHub: https://github.com/ShanmugamRamanathan
LinkedIn: [https://linkedin.com/in/yourprofile ](https://www.linkedin.com/in/shanmugam-ramanathan-260953262/)
