import json
import argparse
import pandas as pd
import numpy as np
from preprocess import clean_skills, parse_date

# ─── Company DNA Classifications ─────────────────────────────────────────────
COMPANY_DNA_MAP = {
    # IT Services (0.60 weight)
    "tcs": "SERVICES", "tata consultancy services": "SERVICES",
    "infosys": "SERVICES", "wipro": "SERVICES", "accenture": "SERVICES",
    "cognizant": "SERVICES", "capgemini": "SERVICES", "hcl": "SERVICES",
    "hcltech": "SERVICES", "mindtree": "SERVICES", "ltts": "SERVICES",
    "lti": "SERVICES", "l&t infotech": "SERVICES", "tech mahindra": "SERVICES",
    "tcs e-serve": "SERVICES",
    
    # FAANG / Big Tech (0.90 weight)
    "google": "FAANG", "microsoft": "FAANG", "amazon": "FAANG",
    "meta": "FAANG", "netflix": "FAANG", "apple": "FAANG",
    "amazon dev centre": "FAANG", "amazon india": "FAANG",
    
    # Product (1.00 weight)
    "swiggy": "PRODUCT", "zepto": "PRODUCT", "razorpay": "PRODUCT",
    "phonepe": "PRODUCT", "paytm": "PRODUCT", "ola": "PRODUCT",
    "zomato": "PRODUCT", "cred": "PRODUCT", "flipkart": "PRODUCT",
    "meesho": "PRODUCT", "groww": "PRODUCT", "inmobi": "PRODUCT",
    "urban company": "PRODUCT", "postman": "PRODUCT", "browserstack": "PRODUCT",
    "hasura": "PRODUCT", "fractal analytics": "PRODUCT", "mu sigma": "PRODUCT",
    
    # Startup (1.05 weight)
    "stealth": "STARTUP", "stealth startup": "STARTUP", "seed stage": "STARTUP",
    "early stage": "STARTUP", "founding team": "STARTUP",
}

def classify_company(company_name):
    if not company_name:
        return "UNKNOWN"
    name_lower = str(company_name).lower().strip()
    # Check exact match
    if name_lower in COMPANY_DNA_MAP:
        return COMPANY_DNA_MAP[name_lower]
    # Check partial match
    for k, v in COMPANY_DNA_MAP.items():
        if k in name_lower:
            return v
    return "UNKNOWN"

def get_company_weight(dna_type):
    weights = {
        "PRODUCT": 1.00,
        "STARTUP": 1.05,
        "FAANG": 0.90,
        "SERVICES": 0.60,
        "RESEARCH": 0.80,
        "UNKNOWN": 0.75, # Neutral fallback
    }
    return weights.get(dna_type, 0.75)

# ─── Feature Processing ───────────────────────────────────────────────────────
def process_candidate_features(cand):
    cid = cand.get("candidate_id")
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})
    
    # Profile & Location
    yoe = profile.get("years_of_experience", 0.0)
    current_title = profile.get("current_title", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    
    # Experience timeline
    career_history = cand.get("career_history", [])
    
    # 1. Company DNA Career Score
    company_weights = []
    total_months = 0
    for job in career_history:
        comp = job.get("company", "")
        duration = job.get("duration_months", 0) or 0
        dna = classify_company(comp)
        wt = get_company_weight(dna)
        company_weights.append(wt * duration)
        total_months += duration
        
    career_dna_score = 0.75 # Default neutral
    if total_months > 0:
        career_dna_score = sum(company_weights) / total_months
        
    # 2. Skill Trust Score
    skills = cand.get("skills", [])
    skill_trusts = []
    for s in skills:
        endorsements = s.get("endorsements", 0) or 0
        duration = s.get("duration_months", 0) or 0
        # Formula: log(1 + endorsements) * (duration / 12)
        trust = np.log1p(endorsements) * (duration / 12.0)
        skill_trusts.append(trust)
        
    skill_trust_score = 0.0
    if skill_trusts:
        skill_trust_score = np.mean(skill_trusts)
        
    # 3. Behavioral signals
    notice_period = signals.get("notice_period_days", 90)
    if notice_period is None:
        notice_period = 90
        
    # Recency of login
    last_active_str = signals.get("last_active_date", "")
    days_since_login = 180 # Default very stale if missing
    if last_active_str:
        last_active = parse_date(last_active_str)
        if last_active:
            # Baseline date is challenge current date: June 2026 (let's use 2026-06-26)
            ref_date = datetime(2026, 6, 26)
            delta = ref_date - last_active
            days_since_login = max(0, delta.days)
            
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate is None:
        response_rate = 0.5
        
    github_score = signals.get("github_activity_score", -1)
    if github_score is None:
        github_score = -1
        
    reloc = signals.get("willing_to_relocate", False)
    
    return {
        "candidate_id": cid,
        "years_of_experience": yoe,
        "current_title": current_title,
        "location": location,
        "country": country,
        "career_dna_score": career_dna_score,
        "skill_trust_score": skill_trust_score,
        "notice_period_days": notice_period,
        "days_since_login": days_since_login,
        "recruiter_response_rate": response_rate,
        "github_score": github_score,
        "willing_to_relocate": reloc,
        "num_skills": len(skills)
    }

def build_feature_matrix(candidates_path, output_path):
    print(f"Reading candidates from {candidates_path}...")
    records = []
    count = 0
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            features = process_candidate_features(cand)
            records.append(features)
            count += 1
            if count % 10000 == 0:
                print(f"Processed {count} candidates...")
                
    df = pd.DataFrame(records)
    print(f"Saving feature matrix of shape {df.shape} to {output_path}...")
    df.to_parquet(output_path, compression="snappy", index=False)
    print("Done building feature matrix.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute candidate features offline.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output", default="features.parquet", help="Path to save output Parquet")
    args = parser.parse_args()
    
    build_feature_matrix(args.candidates, args.output)
