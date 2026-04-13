import numpy as np
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

def run_lda(tokens_list: list, k: int):
    """
    Runs Gensim LDA on the preprocessed token list, returns topics and coherence.
    """
    if not tokens_list:
        return {"coherence": 0.0, "topics": [], "edges": []}
        
    dictionary = corpora.Dictionary(tokens_list)
    # Filter dictionary based on extremities
    dictionary.filter_extremes(no_below=2, no_above=0.8) # Loosened thresholds as real subreddits might be small
    
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    
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
        texts=tokens_list,
        dictionary=dictionary,
        coherence="c_v"
    )
    coherence = coherence_model.get_coherence()
    
    # Compute topic prevalence
    topic_counts = np.zeros(k)
    for doc_bow in corpus:
        doc_topics = model.get_document_topics(doc_bow, minimum_probability=0.0)
        for t_id, prob in doc_topics:
            topic_counts[t_id] += prob
            
    total_weight = topic_counts.sum()
    if total_weight > 0:
        prevalence = (topic_counts / total_weight) * 100
    else:
        prevalence = topic_counts

    topics_out = []
    for topic_id in range(k):
        # We need top 5 words
        words = model.show_topic(topic_id, topn=5)
        label = " / ".join(w for w, _ in words[:3])
        topics_out.append({
            "id": topic_id,
            "label": label,
            "pct": float(round(prevalence[topic_id], 1)),
            "words": [[w, float(round(p, 3))] for w, p in words]
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
