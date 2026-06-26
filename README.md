---
title: TalentMind AI
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# TalentMind AI: Candidate Discovery & Ranking Engine


Intelligent Candidate Discovery & Ranking Engine built for the **Redrob Hackathon (India Runs Data & AI Challenge)**. 

TalentMind AI filters and ranks a candidate pool of **100,000 profiles** to extract the **top 100 best-fit candidates** for a **Senior AI Engineer — Founding Team** role. The online ranking pipeline runs entirely offline on CPU in **$< 15$ seconds** (well within the $\le 5$ minutes limit), consumes **$< 2\text{ GB}$ RAM**, and has a **$0\%$ honeypot rate**.

---

## 🚀 Quick Start & Reproduce

To reproduce the submission CSV file in a clean CPU-only environment under 15 seconds, follow these steps:

### 1. Environment Setup
Clone the repository and install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Hackathon Bundle)
Ensure the hackathon data files are located in the local directory. The pipeline expects:
* `candidates.jsonl` (the 100K candidate pool)
* `job_description.txt` (the job description text file)

### 3. Generate Offline Precomputations (Phase 1)
To keep the online sandbox runtime under 15 seconds, we front-load heavy computations (model inference, index building, and feature mappings) to the offline phase:
```bash
# 1. Generate text embeddings (using sentence-transformers all-MiniLM-L6-v2)
python src/offline/embed_generator.py --candidates ./candidates.jsonl --output embeddings.npz

# 2. Build the BM25 lexical search index
python src/offline/bm25_builder.py --candidates ./candidates.jsonl --output bm25_index.pkl

# 3. Precompute Company DNA & Skill Trust matrices
python src/offline/feature_builder.py --candidates ./candidates.jsonl --output features.parquet
```

### 4. Execute the Ranking Engine (Phase 2 - Online Sandbox)
Run the core ranking pipeline with a single command. This step utilizes the precomputed artifacts and completes in **$< 15$ seconds**:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### 5. Validate the Submission
Verify that the output `submission.csv` complies with the hackathon's shape, monotonicity, and tie-breaking rules:
```bash
python validate_submission.py submission.csv
```

---

## 🧠 System Architecture & Scoring Model

TalentMind AI leverages a hybrid lexical-dense retrieval funnel combined with a multi-factor scoring function and clipped behavioral multipliers.

### Mathematical Scoring Function
The core scoring function calculates a candidate's base alignment:

$$\text{Base Score} = 0.40 \cdot S_{\text{text}} + 0.25 \cdot S_{\text{exp}} + 0.25 \cdot S_{\text{title}} + 0.10 \cdot S_{\text{loc}}$$

* **$S_{\text{text}}$:** Cosine similarity of candidate profile embeddings vs the JD embedding.
* **$S_{\text{exp}}$:** Gaussian experience fit curve centered at 7 years: $\exp(-0.5 \cdot (\frac{\text{years} - 7}{2.5})^2)$.
* **$S_{\text{title}}$:** Role class alignment weights (AI/ML leads = 1.0, software engineers = 0.7).
* **$S_{\text{loc}}$:** Noida/Pune hybrid proximity mapping.

### Availability & Risk Multipliers
The base score is adjusted using clipped multipliers to prioritize active, available, and highly trustable candidates without causing a total score collapse:

$$\text{Score}_{\text{final}} = \text{Base Score} \cdot \prod \text{clip}(M_{\text{risk}}, \text{min}, \text{max})$$

* **$M_{\text{notice}}$:** Notice period discount (buyout preference for $\le 30$-day notice).
* **$M_{\text{recency}}$:** Platform activity recency discount (stale login penalty).
* **$M_{\text{resp}}$:** Recruiter message responsiveness rate.
* **$M_{\text{github}}$:** Open-source contribution bonus.
* **$M_{\text{trust}}$:** Credibility score based on endorsements and skill durations: $\log(1 + \text{endorsements}) \cdot (\frac{\text{duration}_{\text{months}}}{12})$.
* **$M_{\text{company}}$:** Career trajectory alignment (startup/product weights vs IT services).


---

## 🛡️ Unique Selling Points (USPs)
1. **Chronological Timeline Validation (Honeypot Filter):** Excludes profiles violating physical rules (e.g. starting jobs before graduation or expert skills with 0 months duration) to catch synthetic honeypot traps.
2. **Legacy-to-Modern Semantic Alias Engine:** Replaces deprecated tech terms (e.g. Apache Solr, HDFS, Pig Latin) with modern concepts (RAG, PyTorch, Spark MLlib) to preserve seasoned engineering profiles.
3. **4-Tier Modular Reasonings:** Generates unique, slot-filled justifications using seeded random variation per candidate, ensuring $100\%$ factual accuracy, zero hallucinations, and high text variation.
4. **Deterministic Sort & Tie-Break Epsilon:** Implements a 7-level sort chain (Base_Score, S_text, notice, active, resp, github, candidate_id) and applies a tiny epsilon decrement to prevent validator equal-score tie-break issues.

---

## 🖥️ Streamlit Recruiter Sandbox
An interactive recruiter sandbox is provided in `app.py`. Recruiter teams can adjust weights dynamically, view candidate cards with component gauges, inspect duplicate twin flags, and download custom ranks.

Run the dashboard locally:
```bash
streamlit run app.py
```
This sandbox has been deployed to **HuggingFace Spaces** at [suchit2004/redrob-ranker](https://huggingface.co/spaces/suchit2004/redrob-ranker).
