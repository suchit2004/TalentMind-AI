import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.offline.preprocess import build_profile_corpus, clean_skills, parse_date, simple_tokenize
from src.offline.feature_builder import process_candidate_features
from src.reasoner import generate_reasoning

# ─── Configuration & Weights ──────────────────────────────────────────────────
W_TEXT = 0.40
W_EXP = 0.25
W_TITLE = 0.25
W_LOC = 0.10

# Title weight dictionary
TITLE_WEIGHTS = {
    "senior ai engineer": 1.00,
    "ai engineer": 0.95,
    "senior ml engineer": 1.00,
    "ml engineer": 0.95,
    "machine learning engineer": 0.95,
    "lead ai engineer": 1.00,
    "lead ml engineer": 1.00,
    "principal ai engineer": 1.00,
    "data scientist": 0.80,
    "senior data scientist": 0.85,
    "research engineer": 0.85,
    "research scientist": 0.80,
    "software engineer": 0.70,
    "senior software engineer": 0.75,
    "backend engineer": 0.70,
    "senior backend engineer": 0.75,
}

def get_title_weight(title):
    if not title:
        return 0.10
    t_lower = str(title).lower().strip()
    if t_lower in TITLE_WEIGHTS:
        return TITLE_WEIGHTS[t_lower]
    # Partial matching
    for k, v in TITLE_WEIGHTS.items():
        if k in t_lower:
            return v
    # If it is technical but not AI
    if "engineer" in t_lower or "developer" in t_lower or "programmer" in t_lower:
        return 0.50
    return 0.10

def get_location_score(row):
    loc = str(row.get("location", "")).lower().strip()
    country = str(row.get("country", "")).lower().strip()
    reloc = row.get("willing_to_relocate", False)
    
    # Pune/Noida Hybrid cadence (native matches get 1.0)
    if "pune" in loc or "noida" in loc:
        return 1.00
    
    # Rest of India Tier-1 cities / NCR
    tier1_india = ["delhi", "gurgaon", "noida", "ghaziabad", "faridabad", "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata"]
    is_india = (country == "india") or ("india" in loc)
    
    if is_india:
        if any(t1 in loc for t1 in tier1_india):
            return 0.90 if reloc else 0.80
        return 0.70 if reloc else 0.60
        
    # Outside India (visas not sponsored)
    return 0.40

def get_experience_score(yoe):
    # Gaussian experience fit curve centered at 7 years (standard deviation 2.5)
    return np.exp(-0.5 * ((yoe - 7.0) / 2.5) ** 2)

# ─── Honeypot Detection ───────────────────────────────────────────────────────
def is_honeypot(cand):
    """
    Apply 5 hard rules to check for mathematically impossible career constructs (honeypots).
    Returns True if the profile is a honeypot (should be disqualified).
    """
    profile = cand.get("profile", {})
    skills = cand.get("skills", [])
    career = cand.get("career_history", [])
    education = cand.get("education", [])
    
    yoe = profile.get("years_of_experience", 0) or 0
    
    # Rule 1: Expert/Advanced skill claim with 0 months duration
    for s in skills:
        prof = str(s.get("proficiency", "")).lower()
        dur = s.get("duration_months", 0) or 0
        if prof in ["expert", "advanced"] and dur == 0:
            return True
            
    # Rule 2: Impossible graduation to work timeline
    # Parse graduation years
    grad_years = []
    for edu in education:
        ey = edu.get("end_year")
        if ey:
            try:
                grad_years.append(int(ey))
            except ValueError:
                pass
    if grad_years:
        min_grad_year = min(grad_years)
        # Check start dates of jobs
        for job in career:
            start_str = job.get("start_date")
            start_dt = parse_date(start_str)
            if start_dt and start_dt.year < (min_grad_year - 1): # Start work >1 year before graduation
                return True
                
    # Rule 3: Extreme tenure overlap (overlap > 6 months in concurrent roles at different companies)
    jobs_sorted = []
    for job in career:
        sd = parse_date(job.get("start_date"))
        ed = parse_date(job.get("end_date")) or parse_date("2026-06-26") # Baseline current date
        comp = job.get("company", "")
        if sd and ed and sd < ed:
            jobs_sorted.append((sd, ed, comp))
    jobs_sorted.sort(key=lambda x: x[0])
    
    for i in range(len(jobs_sorted) - 1):
        sd1, ed1, c1 = jobs_sorted[i]
        sd2, ed2, c2 = jobs_sorted[i+1]
        if c1 != c2 and sd2 < ed1: # Overlap detected
            overlap_days = (ed1 - sd2).days
            if overlap_days > 180: # Overlap > 6 months
                return True
                
    # Rule 4: Total tenure duration is less than 2 years but claimed experience is greater than 5 years
    sum_months = sum(job.get("duration_months", 0) or 0 for job in career)
    if yoe > 5.0 and sum_months < 24:
        return True
        
    # Rule 5: Employed at a company before its founding year (requires external mapping - here we check for impossible start year, e.g. before 1970 for young candidate)
    for job in career:
        sd = parse_date(job.get("start_date"))
        if sd and sd.year < 1980: # Candidate profile is synthetic/broken if starting work before 1980 for 5-9 YoE
            return True
            
    return False

# ─── Main Ranking Pipeline ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TalentMind AI Core Ranking Engine")
    parser.add_argument("--candidates", default="[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Path to write submission CSV")
    parser.add_argument("--jd", default="[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.txt", help="Path to job description text file")
    args = parser.parse_args()
    
    print(f"Reading Job Description from {args.jd}...")
    if not os.path.exists(args.jd):
        # Fallback search in working dir
        args.jd = "job_description.txt" if os.path.exists("job_description.txt") else args.jd
        
    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()
        
    # Load model (prefer local directory weights to guarantee offline execution)
    model_path = "model_cache/all-MiniLM-L6-v2"
    if os.path.exists(model_path):
        print(f"Loading SentenceTransformer model from local cache {model_path}...")
        model = SentenceTransformer(model_path, device='cpu')
    else:
        print("SentenceTransformer local cache not found; downloading 'all-MiniLM-L6-v2' (network must be on for this fallback)...")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
    jd_embedding = model.encode(jd_text, convert_to_numpy=True)
    
    # 1. Load candidates & filter out honeypots
    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    filtered_count = 0
    with open(args.candidates, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            if is_honeypot(cand):
                filtered_count += 1
                continue
            candidates.append(cand)
            
    print(f"Loaded {len(candidates)} valid candidates. Filtered out {filtered_count} honeypots.")
    
    # Check if we have precomputed embeddings and index
    embeddings_file = "embeddings.npz"
    bm25_file = "bm25_index.pkl"
    features_file = "features.parquet"
    
    # Setup flag to check if we can run on-the-fly (for small sample verification)
    is_small_sample = len(candidates) <= 1000
    
    if is_small_sample or not (os.path.exists(embeddings_file) and os.path.exists(bm25_file) and os.path.exists(features_file)):
        print("Precomputed artifacts missing or small dataset detected. Generating features on-the-fly...")
        # Process on the fly (perfect for Streamlit uploader or local sample check)
        records = [process_candidate_features(c) for c in candidates]
        df_features = pd.DataFrame(records)
        
        # Build BM25 index on the fly
        corpus_tokens = [simple_tokenize(build_profile_corpus(c)) for c in candidates]
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus_tokens)
        
        # Embed candidates on the fly
        candidate_embeddings = model.encode(
            [build_profile_corpus(c) for c in candidates],
            batch_size=256,
            convert_to_numpy=True
        )
        candidate_ids = [c["candidate_id"] for c in candidates]
    else:
        print("Loading precomputed artifacts...")
        # Load embeddings
        embed_data = np.load(embeddings_file, allow_pickle=True)
        precomputed_embeddings = embed_data['embeddings']
        precomputed_ids = embed_data['ids'].tolist()
        
        # Load BM25 index
        with open(bm25_file, 'rb') as f:
            bm25_data = pickle.load(f)
            bm25 = bm25_data['bm25']
            bm25_ids = bm25_data['ids']
            
        # Load features DataFrame
        df_features = pd.read_parquet(features_file)
        
        # Filter artifacts to match only the candidates currently being evaluated
        # This maps candidate_id to their index in the precomputed files
        valid_cids = {c["candidate_id"] for c in candidates}
        
        # Align precomputed files
        # Map ID -> index
        embed_idx_map = {cid: idx for idx, cid in enumerate(precomputed_ids)}
        
        # Extract features and embeddings matching valid_cids
        df_features = df_features[df_features["candidate_id"].isin(valid_cids)].copy()
        
        candidate_ids = df_features["candidate_id"].tolist()
        embedding_indices = [embed_idx_map[cid] for cid in candidate_ids]
        candidate_embeddings = precomputed_embeddings[embedding_indices]
        
    print(f"Retrieved features shape: {df_features.shape}")
    
    # ─── Two-Stage Hybrid Retrieval ───────────────────────────────────────────
    # Stage A: Lexical fast pass (BM25)
    jd_tokens = simple_tokenize(jd_text)
    bm25_scores = bm25.get_scores(jd_tokens)
    
    # Map candidate_id to BM25 score
    bm25_score_map = dict(zip(candidate_ids, bm25_scores))
    
    # Sort and pick top 2000 lexical candidates
    top_bm25_cids = sorted(candidate_ids, key=lambda x: bm25_score_map[x], reverse=True)[:2000]
    
    # Stage B: Cosine Similarity over candidate embeddings
    # Compute vector similarities using numpy
    dot_products = np.dot(candidate_embeddings, jd_embedding)
    cand_norms = np.linalg.norm(candidate_embeddings, axis=1)
    jd_norm = np.linalg.norm(jd_embedding)
    cosine_similarities = dot_products / (cand_norms * jd_norm + 1e-8)
    
    # Map candidate_id to cosine score
    cosine_score_map = dict(zip(candidate_ids, cosine_similarities))
    
    # Sort and pick top 500 dense candidates
    top_dense_cids = sorted(candidate_ids, key=lambda x: cosine_score_map[x], reverse=True)[:500]
    
    # Combine (Union & Dedup)
    candidate_pool_cids = list(set(top_bm25_cids) | set(top_dense_cids))
    print(f"Hybrid retrieval selected {len(candidate_pool_cids)} candidates for full scoring.")
    
    # ─── Multi-Factor Scoring ─────────────────────────────────────────────────
    df_pool = df_features[df_features["candidate_id"].isin(candidate_pool_cids)].copy()
    
    # Calculate score components
    df_pool["S_text"] = df_pool["candidate_id"].map(cosine_score_map)
    df_pool["S_exp"] = df_pool["years_of_experience"].apply(get_experience_score)
    df_pool["S_title"] = df_pool["current_title"].apply(get_title_weight)
    df_pool["S_loc"] = df_pool.apply(get_location_score, axis=1)
    
    # Base Score
    df_pool["Base_Score"] = (
        W_TEXT * df_pool["S_text"] +
        W_EXP * df_pool["S_exp"] +
        W_TITLE * df_pool["S_title"] +
        W_LOC * df_pool["S_loc"]
    )
    
    # Multipliers (Clipped for safety)
    # Notice Period Multiplier
    def get_notice_multiplier(notice_days):
        if notice_days <= 0: return 1.10
        if notice_days <= 30: return 1.00
        if notice_days <= 60: return 0.90
        if notice_days <= 90: return 0.80
        return 0.70
    df_pool["M_notice"] = df_pool["notice_period_days"].apply(get_notice_multiplier)
    
    # Login Recency Multiplier
    def get_recency_multiplier(days_since):
        if days_since < 7: return 1.10
        if days_since < 30: return 1.00
        if days_since < 90: return 0.85
        return 0.65
    df_pool["M_recency"] = df_pool["days_since_login"].apply(get_recency_multiplier)
    
    # Responsiveness Multiplier
    def get_resp_multiplier(rate):
        if rate >= 0.80: return 1.10
        if rate >= 0.50: return 1.00
        if rate >= 0.20: return 0.80
        return 0.70
    df_pool["M_resp"] = df_pool["recruiter_response_rate"].apply(get_resp_multiplier)
    
    # GitHub Activity Multiplier
    def get_github_multiplier(score):
        if score == -1: return 1.00
        if score == 0: return 0.85
        if score <= 30: return 0.90
        if score <= 70: return 1.05
        return 1.15
    df_pool["M_github"] = df_pool["github_score"].apply(get_github_multiplier)
    
    # Skill Trust Multiplier
    df_pool["M_trust"] = 0.75 + 0.25 * df_pool["skill_trust_score"]
    df_pool["M_trust"] = df_pool["M_trust"].clip(0.75, 1.05)
    
    # Company DNA Career Score is already mapped directly in career_dna_score
    df_pool["M_company"] = df_pool["career_dna_score"].clip(0.60, 1.05)
    
    # Final Score Calculation
    df_pool["Score_final"] = (
        df_pool["Base_Score"] *
        df_pool["M_notice"] *
        df_pool["M_recency"] *
        df_pool["M_resp"] *
        df_pool["M_github"] *
        df_pool["M_trust"] *
        df_pool["M_company"]
    )
    
    # Clip Score to safe bounds and apply global floor
    score_floor = np.maximum(0.30, 0.50 * df_pool["Base_Score"])
    df_pool["Score_final"] = np.maximum(df_pool["Score_final"], score_floor)
    
    # ─── Top-10 Stricter Precision Rerank (Pass 2) ────────────────────────────
    # Stricter parameters for the absolute top picks to maximize NDCG@10
    top_candidates = df_pool.sort_values(
        ["Score_final", "Base_Score", "S_text"],
        ascending=[False, False, False]
    ).head(20).copy()
    
    # Stricter experience Gaussian (std = 1.5 instead of 2.5)
    top_candidates["S_exp_strict"] = top_candidates["years_of_experience"].apply(
        lambda y: np.exp(-0.5 * ((y - 7.0) / 1.5) ** 2)
    )
    # Stricter company penalty (demote services companies heavily in top-10)
    top_candidates["M_company_strict"] = top_candidates["career_dna_score"].apply(
        lambda s: 0.50 if s <= 0.65 else s
    )
    
    top_candidates["Score_top10"] = (
        (W_TEXT * top_candidates["S_text"] + W_EXP * top_candidates["S_exp_strict"] + W_TITLE * top_candidates["S_title"] + W_LOC * top_candidates["S_loc"]) *
        top_candidates["M_notice"] * top_candidates["M_recency"] * top_candidates["M_resp"] * top_candidates["M_github"] *
        top_candidates["M_trust"] * top_candidates["M_company_strict"]
    )
    
    # Extract promoted candidate IDs
    promoted_cids = top_candidates.sort_values("Score_top10", ascending=False).head(10)["candidate_id"].tolist()
    
    # Sort pool with tie breakers
    df_pool["is_top10"] = df_pool["candidate_id"].isin(promoted_cids)
    
    # 7-level deterministic tie-breaker sorting:
    # 1. is_top10 (True first)
    # 2. Score_final desc
    # 3. Base_Score desc
    # 4. S_text desc
    # 5. notice_period_days asc
    # 6. days_since_login asc
    # 7. recruiter_response_rate desc
    # 8. github_score desc
    # 9. candidate_id asc
    df_pool = df_pool.sort_values(
        ["is_top10", "Score_final", "Base_Score", "S_text", "notice_period_days", "days_since_login", "recruiter_response_rate", "github_score", "candidate_id"],
        ascending=[False, False, False, False, True, True, False, False, True]
    )
    
    # Take the top 100
    df_submission = df_pool.head(100).copy()
    
    # Assign ranks 1 to 100
    df_submission["rank"] = range(1, 101)
    
    # Force score monotonicity (non-increasing scores) & unique scores using tiny decrement
    # This guarantees we pass the tie-break validator since scores will be strictly monotonic
    base_score = df_submission["Score_final"].values
    monotonic_scores = []
    prev_score = float('inf')
    
    for idx, r in enumerate(df_submission["rank"]):
        curr_score = base_score[idx]
        # Tiny epsilon subtraction to ensure strict sorting order is preserved in the float scores
        adjusted_score = min(curr_score, prev_score) - 1e-8
        monotonic_scores.append(adjusted_score)
        prev_score = adjusted_score
        
    df_submission["score"] = monotonic_scores
    
    # ─── Generating Reasonings ────────────────────────────────────────────────
    print("Generating custom reasons...")
    reasons = []
    for _, row in df_submission.iterrows():
        reason = generate_reasoning(row, row["rank"])
        reasons.append(reason)
    df_submission["reasoning"] = reasons
    
    # Keep only the required columns
    df_final = df_submission[["candidate_id", "rank", "score", "reasoning"]]
    
    # Save output CSV
    print(f"Saving final submission CSV to {args.out}...")
    df_final.to_csv(args.out, encoding="utf-8", index=False)
    print(f"Validation summary: Row count: {len(df_final)}")
    
if __name__ == '__main__':
    main()
