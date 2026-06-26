import json
import argparse
import os
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from src.offline.preprocess import build_profile_corpus
except ModuleNotFoundError:
    from preprocess import build_profile_corpus

def generate_embeddings(candidates_path, output_path, batch_size=256):
    print(f"Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    # Load model on CPU
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    candidate_ids = []
    texts = []
    
    print(f"Reading candidates from {candidates_path}...")
    count = 0
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            corpus = build_profile_corpus(cand)
            
            candidate_ids.append(cid)
            texts.append(corpus)
            count += 1
            
            if count % 10000 == 0:
                print(f"Read {count} candidates...")
                
    print(f"Total candidates read: {count}")
    print("Generating dense embeddings on CPU using multi-process pool (this will utilize all CPU cores)...")
    
    # Start multi-process pool
    pool = model.start_multi_process_pool()
    
    # Run multi-process encoding
    embeddings = model.encode(
        texts,
        pool=pool,
        batch_size=batch_size,
        show_progress_bar=True
    )
    
    # Stop multi-process pool
    model.stop_multi_process_pool(pool)
    
    # Cast to float16 to reduce size by half (from float32)
    embeddings_f16 = embeddings.astype(np.float16)
    
    print(f"Saving embeddings and candidate IDs to {output_path}...")
    np.savez_compressed(output_path, embeddings=embeddings_f16, ids=np.array(candidate_ids))
    print("Done generating embeddings.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dense embeddings offline for candidates.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output", default="embeddings.npz", help="Path to save output npz")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()
    
    generate_embeddings(args.candidates, args.output, args.batch_size)
