import re
from datetime import datetime

def normalize_text(text):
    """Normalize text by converting to lowercase and stripping whitespace."""
    if not text:
        return ""
    return str(text).lower().strip()

def clean_skills(skills_list):
    """Extract and clean names of skills from candidate skills list."""
    if not skills_list:
        return []
    cleaned = []
    for s in skills_list:
        if isinstance(s, dict) and "name" in s:
            name = normalize_text(s["name"])
            if name:
                cleaned.append(name)
        elif isinstance(s, str):
            name = normalize_text(s)
            if name:
                cleaned.append(name)
    return cleaned

def parse_date(date_str):
    """Safely parse date string into a datetime object. Returns None if invalid/null."""
    if not date_str:
        return None
    try:
        # Standard format is YYYY-MM-DD
        return datetime.strptime(str(date_str).strip(), "%Y-%m-%d")
    except ValueError:
        try:
            # Fallback to YYYY
            year = int(str(date_str).strip()[:4])
            return datetime(year, 1, 1)
        except Exception:
            return None

def build_profile_corpus(candidate):
    """
    Concatenate profile fields to form a single corpus text for lexical
    and semantic indexing.
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    
    # Extract skills
    skills = clean_skills(candidate.get("skills", []))
    skills_text = ", ".join(skills)
    
    # Extract career history descriptions
    career_texts = []
    for job in candidate.get("career_history", []):
        desc = job.get("description", "")
        title = job.get("title", "")
        if title:
            career_texts.append(title)
        if desc:
            career_texts.append(desc)
            
    # Concatenate all components
    parts = [
        headline,
        summary,
        current_title,
        skills_text,
        " ".join(career_texts)
    ]
    
    # Filter empty strings and join
    corpus = " ".join([p.strip() for p in parts if p and str(p).strip()])
    return corpus
