import numpy as np
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import requests

def get_llm_topic_label(words, subreddit):
    """
    Queries a local Ollama LLM to generate a short 2-3 word summary label based on the context of the keywords in each topic.
    Falls back to normal keywords string if Ollama is not running.
    """
    keyword_list = [w for w, _ in words]
    prompt = f"Given these frequently occurring keywords from a Reddit topic in r/{subreddit}: {', '.join(keyword_list)}\nProvide a short, concise 2-3 word label that summarizes what this topic is about. Return ONLY the label without punctuation or quotes, no explanation."
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",  # Change this to whatever model you have pulled in Ollama (e.g. 'mistral', 'llama3', 'phi3')
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            },
            timeout=10
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # Clean up the output in case the LLM ignored instructions
            result = result.replace('"', '').replace("'", "")
            if len(result) > 0 and len(result) < 30:
                return result
    except Exception:
        pass # Ignore errors (like connection refused) and fall back
        
    return " / ".join(keyword_list[:3])

def run_lda(tokens_list: list, k: int, subreddit: str = "subreddit"):
    """
    Runs Gensim LDA on the preprocessed token list, returns topics and coherence.
    """
    if not tokens_list:
        return {"coherence": 0.0, "topics": [], "edges": []}
        
    just_tokens = [d["tokens"] if isinstance(d, dict) else d for d in tokens_list]
        
    dictionary = corpora.Dictionary(just_tokens)
    # Filter dictionary based on extremities
    dictionary.filter_extremes(no_below=2, no_above=0.8) # Loosened thresholds as real subreddits might be small
    
    corpus = [dictionary.doc2bow(tokens) for tokens in just_tokens]
    
    if len(corpus) == 0 or len(dictionary) == 0:
        return {"coherence": 0.0, "topics": [], "edges": []}

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=10, 
        iterations=50,
        random_state=42,
        alpha="auto",
        eta="auto",
        per_word_topics=True,
    )
    
    coherence_model = CoherenceModel(
        model=model,
        texts=just_tokens,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    
    # Compute topic prevalence and top docs
    topic_counts = np.zeros(k)
    topic_top_docs = {i: [] for i in range(k)}
    
    for doc_idx, doc_bow in enumerate(corpus):
        doc_topics = model.get_document_topics(doc_bow, minimum_probability=0.0)
        for t_id, prob in doc_topics:
            topic_counts[t_id] += prob
            if prob > 0.2:
                topic_top_docs[t_id].append((prob, doc_idx))
                
    for t_id in range(k):
        topic_top_docs[t_id] = sorted(topic_top_docs[t_id], key=lambda x: x[0], reverse=True)[:3]
            
    total_weight = topic_counts.sum()
    if total_weight > 0:
        prevalence = (topic_counts / total_weight) * 100
    else:
        prevalence = topic_counts

    topics_out = []
    for topic_id in range(k):
        # We need top 5 words
        words = model.show_topic(topic_id, topn=5)
        label = get_llm_topic_label(words, subreddit)
        
        citations = []
        for prob, doc_idx in topic_top_docs[topic_id]:
            doc_obj = tokens_list[doc_idx]
            if isinstance(doc_obj, dict) and doc_obj.get("url"):
                snippet = doc_obj.get("original_text", "")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                citations.append({
                    "url": doc_obj["url"],
                    "text": snippet
                })
                
        topics_out.append({
            "id": topic_id,
            "label": label,
            "pct": float(round(prevalence[topic_id], 1)),
            "words": [[w, float(round(p, 3))] for w, p in words],
            "citations": citations
        })
        
    # Calculate similarity edges using topic-term matrix
    topic_term = model.get_topics() # shape: (k, vocab_size)
    edges = []
    
    for i in range(k):
        for j in range(i + 1, k):
            v1 = topic_term[i]
            v2 = topic_term[j]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(v1, v2) / (norm1 * norm2)
                if sim > 0.1: # Only include edge if similarity is above threshold
                    edges.append([i, j, float(round(sim, 2))])
                    
    return {
        "coherence": float(round(coherence, 2)),
        "topics": topics_out,
        "edges": edges
    }
