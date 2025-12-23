import ollama
from typing import List, Dict

OLLAMA_MODEL = "llama3.2"

def calculate_context_relevancy(question: str, retrieved_docs: List[Dict], model: str = OLLAMA_MODEL) -> float:
    """
    Measures if retrieved KB snippets are actually relevant to the question.
    Returns score between 0 and 1.
    """
    # Combine all retrieved text
    all_text = "\n\n".join([doc['text'] for doc in retrieved_docs])
    
    prompt = f"""Given this question and retrieved documents, identify which sentences are relevant to answering the question.

QUESTION: {question}

RETRIEVED DOCUMENTS:
{all_text}

TASK: List ONLY the sentence numbers that are relevant to answering the question. If a document has no relevant sentences, skip it.

Format: Just list numbers like: 1, 3, 5, 7
If nothing is relevant, respond with: NONE
"""
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    answer = response['message']['content'].strip()
    
    # Count total sentences in retrieved docs
    total_sentences = sum([len([s for s in doc['text'].split('.') if s.strip()]) 
                          for doc in retrieved_docs])
    
    # Count relevant sentences
    if answer.upper() == "NONE":
        relevant_count = 0
    else:
        # Extract numbers from response
        import re
        numbers = re.findall(r'\d+', answer)
        relevant_count = len(numbers)
    
    if total_sentences == 0:
        return 0.0
    
    score = min(relevant_count / total_sentences, 1.0)
    return round(score, 3)


def calculate_answer_completeness(question: str, retrieved_docs: List[Dict], model: str = OLLAMA_MODEL) -> Dict:
    """
    Breaks question into sub-components and checks what % can be answered.
    Returns score and detailed breakdown.
    """
    all_text = "\n\n".join([doc['text'] for doc in retrieved_docs])
    
    prompt = f"""You are evaluating if retrieved KB documents can completely answer a user's question.

QUESTION: {question}

RETRIEVED KB DOCUMENTS:
{all_text}

TASK:
1. Break the question into 3-5 sub-questions that need to be answered
2. For each sub-question, determine if the KB documents provide an answer
3. Provide a completeness score

Format your response EXACTLY like this:
SUB-QUESTIONS:
1. [sub-question] - [ANSWERED/PARTIAL/NOT_ANSWERED]
2. [sub-question] - [ANSWERED/PARTIAL/NOT_ANSWERED]
3. [sub-question] - [ANSWERED/PARTIAL/NOT_ANSWERED]

COMPLETENESS_SCORE: X.XX (0.0 to 1.0)
"""
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    answer = response['message']['content']
    
    # Extract score using regex
    import re
    score_match = re.search(r'COMPLETENESS_SCORE:\s*([\d.]+)', answer)
    score = float(score_match.group(1)) if score_match else 0.5
    
    return {
        'score': round(score, 3),
        'breakdown': answer
    }


def calculate_faithfulness(evaluation_text: str, retrieved_docs: List[Dict], model: str = OLLAMA_MODEL) -> float:
    """
    Checks if the evaluation/answer is faithful to source documents (no hallucination).
    Returns score between 0 and 1.
    """
    all_text = "\n\n".join([doc['text'] for doc in retrieved_docs])
    
    prompt = f"""You are checking if an AI's response is faithful to source documents (no hallucination).

AI'S EVALUATION/RESPONSE:
{evaluation_text}

SOURCE KB DOCUMENTS:
{all_text}

TASK:
1. Extract all factual claims/statements from the AI's response
2. For each claim, check if it can be verified in the source documents
3. Calculate faithfulness score

Format your response EXACTLY like this:
CLAIMS:
1. [claim] - [VERIFIED/NOT_VERIFIED]
2. [claim] - [VERIFIED/NOT_VERIFIED]

FAITHFULNESS_SCORE: X.XX (0.0 to 1.0)
"""
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    answer = response['message']['content']
    
    # Extract score
    import re
    score_match = re.search(r'FAITHFULNESS_SCORE:\s*([\d.]+)', answer)
    score = float(score_match.group(1)) if score_match else 0.8
    
    return round(score, 3)


def calculate_precision_at_k(question: str, retrieved_docs: List[Dict], k: int = 3, model: str = OLLAMA_MODEL) -> Dict:
    """
    Measures how many of the top-K retrieved documents are actually relevant.
    Returns precision score and per-doc relevance.
    """
    # Only evaluate top-k documents
    top_k_docs = retrieved_docs[:k]
    
    relevance_results = []
    
    for i, doc in enumerate(top_k_docs, 1):
        prompt = f"""Is this KB document relevant for answering the user's question?

QUESTION: {question}

KB DOCUMENT:
{doc['text']}

Respond with ONLY one word: RELEVANT or NOT_RELEVANT
"""
        
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        answer = response['message']['content'].strip().upper()
        is_relevant = "RELEVANT" in answer and "NOT_RELEVANT" not in answer
        
        relevance_results.append({
            'doc_num': i,
            'score': doc['score'],
            'relevant': is_relevant
        })
    
    relevant_count = sum([1 for r in relevance_results if r['relevant']])
    precision = relevant_count / k
    
    return {
        'precision_score': round(precision, 3),
        'relevant_count': relevant_count,
        'total_evaluated': k,
        'details': relevance_results
    }


def evaluate_kb_quality_with_metrics(question: str, retrieved_docs: List[Dict], llm_evaluation: str) -> Dict:
    """
    Main function that calculates all metrics for a question.
    """
    print("\n📊 Calculating Advanced Metrics...")
    
    # 1. Context Relevancy
    print("   ⏳ Context Relevancy...")
    context_relevancy = calculate_context_relevancy(question, retrieved_docs)
    
    # 2. Answer Completeness
    print("   ⏳ Answer Completeness...")
    completeness_result = calculate_answer_completeness(question, retrieved_docs)
    
    # 3. Faithfulness
    print("   ⏳ Faithfulness...")
    faithfulness = calculate_faithfulness(llm_evaluation, retrieved_docs)
    
    # 4. Precision@K
    print("   ⏳ Precision@K...")
    precision_result = calculate_precision_at_k(question, retrieved_docs, k=3)
    
    metrics = {
        'context_relevancy': context_relevancy,
        'answer_completeness': completeness_result['score'],
        'completeness_breakdown': completeness_result['breakdown'],
        'faithfulness': faithfulness,
        'precision_at_3': precision_result['precision_score'],
        'precision_details': precision_result['details']
    }
    
    print("   ✅ Metrics calculated!\n")
    
    return metrics