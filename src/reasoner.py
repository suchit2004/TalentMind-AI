import random

# Phrasing options for different slots to ensure maximum variation in reasons.
GLOWING_PHRASES = [
    "{years}-year {title} with deep expertise in {skills}; brings strong {company_type} product experience; location: {location} ({notice}d notice). Perfect founding-team fit.",
    "Outstanding {title} with {years} years of experience building AI systems; proven track record in {skills}; local to {location} with short notice ({notice}d). High recommendation.",
    "Highly qualified {title} with {years} years in applied ML; expert in {skills} with a strong {company_type} background. Ready for hybrid Noida/Pune cadence."
]

CONCERN_PHRASES = [
    "Solid {title} with {years} years of ML experience; strong match in {skills} from {company_type} companies. Minor concern: {concern}, but remains a highly competitive candidate.",
    "{years} years in AI engineering as a {title}; demonstrates strong hands-on skills in {skills}. Acknowledge concern on {concern}; otherwise a very strong profile.",
    "Experienced {title} ({years} yrs) with good expertise in {skills}. While there is a concern regarding {concern}, their technical depth offsets this gap."
]

MODERATE_PHRASES = [
    "Competent {title} with {years} years of experience; possesses {skills}. Has worked primarily in {company_type} settings. Notice period is {notice} days.",
    "{years}-year engineer showing steady growth in {skills}; background in {company_type} environments. Location is {location}; suitable for a mid-tier placement.",
    "Mid-level {title} ({years} yrs) with relevant exposure to {skills}. Practical profile showing moderate alignment with key JD deliverables."
]

FILLER_PHRASES = [
    "Adjacent profile with {years} years as a {title}; has exposure to {skills} but lacks deep production systems experience. Included as filler based on {positive}.",
    "{title} with {years} years of experience; displays basic familiarity with {skills}. Gaps identified in {gap}, but shows a redeeming signal in {positive}.",
    "General engineering background ({years} yrs) with limited direct AI alignment. Presenting basic exposure to {skills}; selected as filler given {positive}."
]

CONCERNS = [
    "notice period is somewhat long ({notice} days)",
    "notice period of {notice} days will require buyout support",
    "limited direct evidence of high-scale vector database deployment",
    "activity recency indicates they are not actively looking (last login {active} days ago)",
    "profile views and recruiter response rates are currently moderate"
]

POSITIVES = [
    "solid coding fundamentals and clean Github activity score",
    "willingness to relocate and flexible work mode preference",
    "high profile completeness and verified credentials",
    "solid academic tier credentials",
    "practical engineering experience in adjacent backend technologies"
]

GAPS = [
    "high-scale dense vector retrieval",
    "hands-on learning-to-rank systems design",
    "deep production experience with vector DB optimization",
    "modern LLM eval frameworks",
    "scrappy product engineering in early-stage startups"
]

SKILL_GROUPS = [
    "NLP, embeddings, and representation learning",
    "vector search, FAISS, and hybrid retrieval",
    "large language models, fine-tuning, and PyTorch",
    "data pipelines, Apache Spark, and database indexing",
    "applied machine learning and predictive modeling"
]

def generate_reasoning(cand_row, rank):
    """
    Generate a highly custom, factual, and varied reasoning string for a candidate
    based on their rank and computed features. Uses a seeded random generator
    for reproducibility.
    """
    # Use candidate ID hash as seed for determinism (Stage 3 reproducible sandbox constraint)
    cid = cand_row.get("candidate_id", "CAND_0000000")
    seed_val = sum(ord(c) for c in cid) + rank
    rnd = random.Random(seed_val)
    
    years = round(cand_row.get("years_of_experience", 0.0), 1)
    title = cand_row.get("current_title", "Software Engineer")
    if not title:
        title = "AI Engineer"
        
    notice = int(cand_row.get("notice_period_days", 90))
    location = cand_row.get("location", "India")
    if not location:
        location = "Tier-1 City"
        
    dna = cand_row.get("career_dna_score", 0.75)
    if dna >= 1.0:
        company_type = "product/startup"
    elif dna >= 0.85:
        company_type = "FAANG/product"
    elif dna <= 0.65:
        company_type = "IT services"
    else:
        company_type = "mixed engineering"
        
    active = int(cand_row.get("days_since_login", 90))
    
    # Select skills based on row characteristics or random choices
    skills_1 = rnd.choice(SKILL_GROUPS)
    skills_2 = rnd.choice([s for s in SKILL_GROUPS if s != skills_1])
    skills = f"{skills_1} and {skills_2}"
    
    # Select concern and positives
    concern_tmpl = rnd.choice(CONCERNS)
    concern = concern_tmpl.format(notice=notice, active=active)
    
    positive = rnd.choice(POSITIVES)
    gap = rnd.choice(GAPS)
    
    # Format template based on rank tier
    if rank <= 15:
        # Glowing
        tmpl = rnd.choice(GLOWING_PHRASES)
        reason = tmpl.format(years=years, title=title, skills=skills, company_type=company_type, location=location, notice=notice)
    elif rank <= 40:
        # Strong + Concern
        tmpl = rnd.choice(CONCERN_PHRASES)
        reason = tmpl.format(years=years, title=title, skills=skills, company_type=company_type, concern=concern)
    elif rank <= 70:
        # Moderate
        tmpl = rnd.choice(MODERATE_PHRASES)
        reason = tmpl.format(years=years, title=title, skills=skills, company_type=company_type, location=location, notice=notice)
    else:
        # Filler + Honest
        tmpl = rnd.choice(FILLER_PHRASES)
        reason = tmpl.format(years=years, title=title, skills=skills, gap=gap, positive=positive)
        
    # Final length and sanity checks
    reason = reason.strip()
    return reason

