import io
import csv
import traceback
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# Import pipeline functions
from pipeline.collect import collect_data
from pipeline.preprocess import preprocess_data
from pipeline.model import run_lda

app = Flask(__name__)
# Enable CORS to allow Electron renderer to fetch easily
CORS(app)

# In-memory storage for the latest run result
latest_result = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/run", methods=["POST"])
def run_pipeline():
    global latest_result
    
    data = request.json
    subreddit = data.get("subreddit", "MohsSurgery")
    # Clean up 'r/' if user provided it
    if subreddit.startswith("r/"):
        subreddit = subreddit[2:]
        
    k = data.get("k", 10)
    date_from = data.get("date_from")
    date_to = data.get("date_to")
    
    if not all([subreddit, k, date_from, date_to]):
        return jsonify({"error": "Missing required parameters: subreddit, k, date_from, date_to"}), 400
        
    try:
        # Step 1: Collect Data
        theme_desc, raw_texts = collect_data(subreddit, date_from, date_to)
        if not raw_texts:
            return jsonify({"error": f"No data found for r/{subreddit} in the given date range."}), 404
            
        # Step 2: Preprocess
        tokens_list = preprocess_data(raw_texts)
        if not tokens_list:
            return jsonify({"error": "Not enough valid text data remaining after preprocessing."}), 400
            
        # Step 3: Train LDA
        if k == "auto":
            best_coherence = -1
            best_results = None
            best_k = 10
            for candidate_k in [5, 8, 12, 16]:
                try:
                    res = run_lda(tokens_list, int(candidate_k), subreddit)
                    if res["coherence"] > best_coherence:
                        best_coherence = res["coherence"]
                        best_results = res
                        best_k = candidate_k
                except Exception as e:
                    print(f"Error testing k={candidate_k}: {e}")
            
            if not best_results:
                return jsonify({"error": "Failed to auto-optimize topics."}), 500
                
            model_results = best_results
            k = best_k
        else:
            k = int(k)
            model_results = run_lda(tokens_list, k, subreddit)
        
        # Step 4: Finalize Output
        response_data = {
            "subreddit": subreddit,
            "theme": theme_desc,
            "k": k,
            "coherence": model_results["coherence"],
            "topics": model_results["topics"],
            "edges": model_results["edges"]
        }
        
        # Save to memory for potential CSV export later
        latest_result = response_data
        
        return jsonify(response_data)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/export", methods=["GET"])
def export_csv():
    global latest_result
    if not latest_result:
        return jsonify({"error": "No data to export"}), 404
        
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Rank", "Topic_ID", "Label", "Prevalence_Pct", "Top_Words"])
    
    topics = latest_result["topics"]
    # Sort topics by prevalence for the CSV output
    topics_sorted = sorted(topics, key=lambda x: x["pct"], reverse=True)
    
    for i, t in enumerate(topics_sorted):
        word_list = [w[0] for w in t["words"]]
        words_str = ", ".join(word_list)
        writer.writerow([
            i + 1, 
            t["id"], 
            t["label"], 
            t["pct"], 
            words_str
        ])
        
    output = make_response(si.getvalue())
    
    # We send normal header; however, in Electron we might handle fetching this string manually 
    # instead of standard browser download, but we include it nicely just in case.
    filename = f"topics_r{latest_result['subreddit']}_k{latest_result['k']}.csv"
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    
    return output

if __name__ == "__main__":
    # Electron spawns this locally on port 5173
    app.run(host="127.0.0.1", port=5173, debug=False)
