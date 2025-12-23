import os
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch

# Config
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "infra_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"

def retrieve_kb_snippets(question: str, top_k: int = 3):
    """Retrieve most relevant KB snippets for a question."""
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    
    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Encode question
    q_vec = model.encode([question])[0].tolist()
    
    # Search Qdrant
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=top_k,
    )
    
    # Extract snippets and scores
    results = []
    for point in search_result.points:
        results.append({
            'text': point.payload['text'],
            'score': point.score
        })
    
    return results

def evaluate_kb_quality(question: str, snippets: list):
    """Use Ollama to evaluate if KB snippets can answer the question."""
    
    # Build context from retrieved snippets
    context = "\n\n".join([f"[Snippet {i+1}] (Score: {s['score']:.3f})\n{s['text']}" 
                           for i, s in enumerate(snippets)])
    
    # Prompt for LLM evaluation
    prompt = f"""You are a knowledge base quality evaluator for IT infrastructure troubleshooting documentation.

USER QUESTION:
{question}

RETRIEVED KB ARTICLES:
{context}

TASK:
Evaluate if the retrieved KB articles contain sufficient information to answer the user's question completely.

Provide your evaluation in this format:
VERDICT: [Complete/Partial/Missing]
CONFIDENCE: [High/Medium/Low]
EXPLANATION: [Brief explanation of why this verdict]
GAPS: [What information is missing, if any]
"""

    # Call Ollama
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    
    return response['message']['content']

def check_question(question: str):
    """Main function to check KB quality for a question."""
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    
    # Step 1: Retrieve relevant snippets
    print("🔍 Retrieving relevant KB articles...")
    snippets = retrieve_kb_snippets(question, top_k=3)
    
    print(f"✅ Retrieved {len(snippets)} snippets\n")
    for i, s in enumerate(snippets, 1):
        print(f"Snippet {i} - Score: {s['score']:.3f}")
        print(f"{s['text'][:150]}...\n")
    
    # Step 2: Evaluate with LLM
    print("🤖 Evaluating KB quality with Ollama...\n")
    evaluation = evaluate_kb_quality(question, snippets)
    
    print(f"{'='*80}")
    print("EVALUATION RESULT:")
    print(f"{'='*80}")
    print(evaluation)
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Test questions
    test_questions = [
        "Why is my SCOM agent not sending heartbeats?",
        "How do I troubleshoot high CPU usage on a monitored server?",
        "What should I do if SCOM database backup fails?",
    ]
    
    for question in test_questions:
        check_question(question)
        print("\n" + "="*80 + "\n")