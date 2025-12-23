import os
import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
import json
from datetime import datetime
from evaluation_metrics import evaluate_kb_quality_with_metrics

# ---------- Config ----------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "infra_kb"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"
# ----------------------------


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


def check_question_with_metrics(question: str):
    """Enhanced version with advanced metrics."""
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
    
    # Step 2: Basic LLM Evaluation
    print("🤖 Evaluating KB quality with Ollama...\n")
    evaluation = evaluate_kb_quality(question, snippets)
    
    print(f"{'='*80}")
    print("BASIC EVALUATION:")
    print(f"{'='*80}")
    print(evaluation)
    print(f"{'='*80}\n")
    
    # Step 3: Calculate Advanced Metrics
    metrics = evaluate_kb_quality_with_metrics(question, snippets, evaluation)
    
    # Step 4: Display Metrics
    print(f"{'='*80}")
    print("📊 ADVANCED METRICS:")
    print(f"{'='*80}")
    print(f"Context Relevancy:     {metrics['context_relevancy']:.3f} {get_score_emoji(metrics['context_relevancy'])}")
    print(f"Answer Completeness:   {metrics['answer_completeness']:.3f} {get_score_emoji(metrics['answer_completeness'])}")
    print(f"Faithfulness:          {metrics['faithfulness']:.3f} {get_score_emoji(metrics['faithfulness'])}")
    print(f"Precision@3:           {metrics['precision_at_3']:.3f} {get_score_emoji(metrics['precision_at_3'])}")
    print(f"{'='*80}\n")
    
    # Calculate overall quality score (weighted average)
    overall_score = (
        metrics['context_relevancy'] * 0.25 +
        metrics['answer_completeness'] * 0.35 +
        metrics['faithfulness'] * 0.20 +
        metrics['precision_at_3'] * 0.20
    )
    
    print(f"📈 OVERALL KB QUALITY SCORE: {overall_score:.3f} {get_score_emoji(overall_score)}")
    print(f"{'='*80}\n")
    
    # Return full results
    return {
        'question': question,
        'retrieved_snippets': snippets,
        'evaluation': evaluation,
        'metrics': metrics,
        'overall_score': round(overall_score, 3),
        'timestamp': datetime.now().isoformat()
    }


def get_score_emoji(score: float) -> str:
    """Visual indicator for scores."""
    if score >= 0.8:
        return "🟢 Excellent"
    elif score >= 0.6:
        return "🟡 Good"
    elif score >= 0.4:
        return "🟠 Fair"
    else:
        return "🔴 Poor"


def generate_summary_report(all_results: list):
    """Generate aggregate statistics across all test questions."""
    print(f"\n{'='*80}")
    print("📊 SUMMARY REPORT")
    print(f"{'='*80}\n")
    
    total_questions = len(all_results)
    
    # Calculate averages
    avg_context_relevancy = sum([r['metrics']['context_relevancy'] for r in all_results]) / total_questions
    avg_completeness = sum([r['metrics']['answer_completeness'] for r in all_results]) / total_questions
    avg_faithfulness = sum([r['metrics']['faithfulness'] for r in all_results]) / total_questions
    avg_precision = sum([r['metrics']['precision_at_3'] for r in all_results]) / total_questions
    avg_overall = sum([r['overall_score'] for r in all_results]) / total_questions
    
    print(f"Total Questions Tested: {total_questions}")
    print(f"\nAverage Metrics:")
    print(f"  Context Relevancy:   {avg_context_relevancy:.3f}")
    print(f"  Answer Completeness: {avg_completeness:.3f}")
    print(f"  Faithfulness:        {avg_faithfulness:.3f}")
    print(f"  Precision@3:         {avg_precision:.3f}")
    print(f"  Overall Score:       {avg_overall:.3f} {get_score_emoji(avg_overall)}")
    
    # Identify worst performing questions
    print(f"\n🔴 Questions Needing Most Improvement:")
    sorted_results = sorted(all_results, key=lambda x: x['overall_score'])
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {result['question'][:60]}... (Score: {result['overall_score']:.3f})")
    
    print(f"\n{'='*80}\n")
    
    return {
        'total_questions': total_questions,
        'avg_metrics': {
            'context_relevancy': round(avg_context_relevancy, 3),
            'answer_completeness': round(avg_completeness, 3),
            'faithfulness': round(avg_faithfulness, 3),
            'precision_at_3': round(avg_precision, 3),
            'overall_score': round(avg_overall, 3)
        },
        'worst_questions': [
            {'question': r['question'], 'score': r['overall_score']}
            for r in sorted_results[:3]
        ]
    }


if __name__ == "__main__":
    # Test questions
    test_questions = [
        "Why is my SCOM agent not sending heartbeats?",
        "How do I troubleshoot high CPU usage on a monitored server?",
        "What should I do if SCOM database backup fails?",
    ]
    
    print("\n" + "="*80)
    print("🚀 INFRA KB QUALITY CHECKER")
    print("="*80)
    print(f"Testing {len(test_questions)} questions against knowledge base...")
    print("="*80 + "\n")
    
    all_results = []
    
    for question in test_questions:
        result = check_question_with_metrics(question)
        all_results.append(result)
    
    # Generate summary
    summary = generate_summary_report(all_results)
    
    # Save results to JSON
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"kb_quality_detailed_{timestamp}.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Save summary report
    summary_file = os.path.join(output_dir, f"kb_quality_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Detailed results saved to: {detailed_file}")
    print(f"✅ Summary report saved to: {summary_file}")
    print(f"\n{'='*80}")
    print("✨ Quality check complete!")
    print("="*80 + "\n")