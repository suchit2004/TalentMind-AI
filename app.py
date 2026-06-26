import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from sentence_transformers import SentenceTransformer
from rank import get_experience_score, get_title_weight, get_location_score, is_honeypot
from src.offline.preprocess import build_profile_corpus, simple_tokenize
from src.offline.feature_builder import process_candidate_features
from src.reasoner import generate_reasoning

# ─── Page Config & Premium Aesthetics ─────────────────────────────────────────
st.set_page_config(
    page_title="TalentMind AI — Candidate Discovery Sandbox",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stSlider > div > div > div > div {
        background-color: #2563EB;
    }
    .card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #38BDF8;
    }
    .badge-honeypot {
        background-color: #EF4444;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-valid {
        background-color: #10B981;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 TalentMind AI")
st.subheader("Intelligent Candidate Discovery & Ranking Sandbox")

st.markdown("""
This interactive recruiter sandbox allows you to upload candidate profiles, paste a job description, 
adjust the scoring model weights dynamically, and download the validated Stage-compliant CSV.
""")

# ─── Sidebar Configuration ────────────────────────────────────────────────────
st.sidebar.header("🎯 Model Weight Customization")

w_text = st.sidebar.slider("Semantic Match Weight (w1)", 0.0, 1.0, 0.40, 0.05)
w_exp = st.sidebar.slider("Experience Curve Weight (w2)", 0.0, 1.0, 0.25, 0.05)
w_title = st.sidebar.slider("Title Fit Weight (w3)", 0.0, 1.0, 0.25, 0.05)
w_loc = st.sidebar.slider("Location Match Weight (w4)", 0.0, 1.0, 0.10, 0.05)

# Normalization check
total_weight = w_text + w_exp + w_title + w_loc
if abs(total_weight - 1.0) > 1e-4:
    st.sidebar.warning(f"Total weight is {total_weight:.2f} (should sum to 1.0 for standard calibration).")
else:
    st.sidebar.success("Weights are calibrated correctly (sums to 1.0).")

# ─── File Uploads & Input ─────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1. Upload Candidate Sample")
    uploaded_file = st.file_uploader("Upload candidates file (.json or .jsonl)", type=["json", "jsonl"])
    
    # Load sample default if not uploaded
    if uploaded_file is None:
        st.info("No file uploaded. Please upload a candidates JSON subset to begin.")
        
with col2:
    st.markdown("### 2. Job Description")
    jd_input = st.text_area(
        "Job Description Text",
        value="Job Description: Senior AI Engineer — Founding Team\nCompany: Redrob AI\nLocation: Pune/Noida, India\nExperience Required: 5–9 years\nOwn the intelligence layer: embeddings, retrieval, ranking, vector databases (Qdrant, Pinecone), hybrid search, and evaluation frameworks (NDCG, MAP).",
        height=180
    )

# ─── Processing ───────────────────────────────────────────────────────────────
if uploaded_file is not None and jd_input:
    # Read files
    try:
        if uploaded_file.name.endswith(".json"):
            candidates_data = json.load(uploaded_file)
            if not isinstance(candidates_data, list):
                candidates_data = [candidates_data]
        else: # jsonl
            candidates_data = []
            for line in uploaded_file:
                if line.strip():
                    candidates_data.append(json.loads(line))
        
        st.success(f"Successfully loaded {len(candidates_data)} candidates.")
        
        # ─── Load Model & Embed on-the-fly ────────────────────────────────────
        with st.spinner("Initializing Model & Calculating Similarities..."):
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            jd_emb = model.encode(jd_input, convert_to_numpy=True)
            
            records = []
            for c in candidates_data:
                is_hp = is_honeypot(c)
                features = process_candidate_features(c)
                corpus = build_profile_corpus(c)
                emb = model.encode(corpus, convert_to_numpy=True)
                
                # Cosine Similarity
                cos_sim = np.dot(emb, jd_emb) / (np.linalg.norm(emb) * np.linalg.norm(jd_emb) + 1e-8)
                
                features["is_honeypot"] = is_hp
                features["S_text"] = float(cos_sim)
                records.append(features)
                
            df = pd.DataFrame(records)
            
        # ─── Apply Weights & Multipliers ──────────────────────────────────────
        df["S_exp"] = df["years_of_experience"].apply(get_experience_score)
        df["S_title"] = df["current_title"].apply(get_title_weight)
        df["S_loc"] = df.apply(get_location_score, axis=1)
        
        # Base Score
        df["Base_Score"] = (
            w_text * df["S_text"] +
            w_exp * df["S_exp"] +
            w_title * df["S_title"] +
            w_loc * df["S_loc"]
        )
        
        # Multipliers (Clipped)
        def get_notice_mult(days):
            if days <= 0: return 1.10
            if days <= 30: return 1.00
            if days <= 60: return 0.90
            if days <= 90: return 0.80
            return 0.70
            
        def get_active_mult(days):
            if days < 7: return 1.10
            if days < 30: return 1.00
            if days < 90: return 0.85
            return 0.65
            
        def get_resp_mult(rate):
            if rate >= 0.80: return 1.10
            if rate >= 0.50: return 1.00
            if rate >= 0.20: return 0.80
            return 0.70
            
        def get_github_mult(score):
            if score == -1: return 1.00
            if score == 0: return 0.85
            if score <= 30: return 0.90
            if score <= 70: return 1.05
            return 1.15
            
        df["M_notice"] = df["notice_period_days"].apply(get_notice_mult)
        df["M_recency"] = df["days_since_login"].apply(get_active_mult)
        df["M_resp"] = df["recruiter_response_rate"].apply(get_resp_mult)
        df["M_github"] = df["github_score"].apply(get_github_mult)
        df["M_trust"] = (0.75 + 0.25 * df["skill_trust_score"]).clip(0.75, 1.05)
        df["M_company"] = df["career_dna_score"].clip(0.60, 1.05)
        
        df["Score_final"] = (
            df["Base_Score"] *
            df["M_notice"] *
            df["M_recency"] *
            df["M_resp"] *
            df["M_github"] *
            df["M_trust"] *
            df["M_company"]
        )
        
        # Floor
        floor = np.maximum(0.30, 0.50 * df["Base_Score"])
        df["Score_final"] = np.maximum(df["Score_final"], floor)
        
        # Apply strict honeypot filter for the final ranks (Honeypots score 0.0 or get dropped)
        df.loc[df["is_honeypot"] == True, "Score_final"] = 0.0
        
        # Sort & Rank
        df = df.sort_values(
            ["Score_final", "Base_Score", "candidate_id"],
            ascending=[False, False, True]
        )
        
        # Monotonic decrement adjust for equal scores
        scores = df["Score_final"].values
        monotonic_scores = []
        prev = float('inf')
        for idx, s in enumerate(scores):
            adj = min(s, prev) - 1e-8
            monotonic_scores.append(adj)
            prev = adj
        df["Score_final"] = monotonic_scores
        
        df["rank"] = range(1, len(df) + 1)
        
        # Generate Reasonings
        reasons = [generate_reasoning(row, row["rank"]) for _, row in df.iterrows()]
        df["reasoning"] = reasons
        
        # ─── Dashboard Visualizations ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Pool Overview & Stats")
        
        c_stats1, c_stats2, c_stats3 = st.columns(3)
        with c_stats1:
            total_c = len(df)
            st.metric("Total Candidates Evaluated", total_c)
        with c_stats2:
            hp_count = int(df["is_honeypot"].sum())
            st.metric("Honeypots Detected & Disqualified", hp_count, delta=f"{hp_count/total_c*100:.1f}% of pool", delta_color="inverse")
        with c_stats3:
            avg_score = float(df[df["is_honeypot"] == False]["Score_final"].mean())
            st.metric("Average Score (Valid Pool)", f"{avg_score:.3f}")
            
        # Top 10 Distribution Chart
        st.markdown("#### Score Component Distribution (Top 15 Candidates)")
        top_15 = df[df["is_honeypot"] == False].head(15)
        
        fig = px.bar(
            top_15,
            x="candidate_id",
            y=["S_text", "S_exp", "S_title", "S_loc"],
            title="Individual Score Component Weights",
            labels={"value": "Component Score", "candidate_id": "Candidate ID"},
            barmode="stack",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ─── Candidate Cards ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏆 Detailed Rankings (Top 100 Shortlist)")
        
        for _, row in df.head(100).iterrows():
            cid = row["candidate_id"]
            rnk = row["rank"]
            score_val = row["Score_final"]
            reason = row["reasoning"]
            is_hp = row["is_honeypot"]
            
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:20px; font-weight:bold; color:#F1F5F9;">Rank #{rnk} — {cid}</span>
                            {" <span class='badge-honeypot'>Honeypot Disqualified</span>" if is_hp else " <span class='badge-valid'>Verified Candidate</span>"}
                        </div>
                        <div class="metric-value">{score_val:.4f}</div>
                    </div>
                    <div style="margin-top:8px; color:#94A3B8; font-style:italic;">
                        "{reason}"
                    </div>
                    <div style="margin-top:12px; font-size:13px; color:#CBD5E1;">
                        <b>Experience:</b> {row['years_of_experience']:.1f} years | 
                        <b>Location:</b> {row['location']} | 
                        <b>DNA Score:</b> {row['career_dna_score']:.2f} | 
                        <b>Notice Period:</b> {row['notice_period_days']} days | 
                        <b>GitHub Score:</b> {row['github_score']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # ─── CSV Download ─────────────────────────────────────────────────────
        df_csv = df.head(100)[["candidate_id", "rank", "Score_final", "reasoning"]].rename(columns={"Score_final": "score"})
        csv_data = df_csv.to_csv(encoding="utf-8", index=False)
        
        st.sidebar.markdown("---")
        st.sidebar.download_button(
            label="💾 Download Rankings (CSV)",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv"
        )
        st.sidebar.info("Download matches the exact structure required by validate_submission.py.")
        
    except Exception as e:
        st.error(f"Error parsing candidates file: {e}")
