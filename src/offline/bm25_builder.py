import json
import argparse
import pickle
import re
from preprocess import build_profile_corpus

def simple_tokenize(text):
    """Simple tokenizer to split text into words, lowercase them, and remove non-alphanumeric words."""
    if not text:
        return []
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove single characters and stop-word like tokens for basic noise reduction
    return [w for w in words if len(w) > 1]

def build_bm25_index(candidates_path, output_path):
    print(f"Reading candidates from {candidates_path}...")
    corpus_tokens = []
    candidate_ids = []
    
    count = 0
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            corpus = build_profile_corpus(cand)
            tokens = simple_tokenize(corpus)
            
            corpus_tokens.append(tokens)
            candidate_ids.append(cid)
            count += 1
            
            if count % 10000 == 0:
                print(f"Read and tokenized {count} candidates...")
                
    print(f"Total tokenized candidates: {count}")
    print("Building BM25Okapi index...")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(corpus_tokens)
    
    # Save the BM25 index along with candidate IDs mapping
    data_to_save = {
        "bm25": bm25,
        "ids": candidate_ids
    }
    
    print(f"Serializing and saving BM25 index to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done building BM25 index.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build and serialize BM25 index offline.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output", default="bm25_index.pkl", help="Path to save serialized BM25 index")
    args = parser.parse_args()
    
    build_bm25_index(args.candidates, args.output)
