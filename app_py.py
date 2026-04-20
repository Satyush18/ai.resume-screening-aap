# -*- coding: utf-8 -*-

import streamlit as st
import PyPDF2
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")
st.markdown("### Created by: Satyush Mohapatra")
st.markdown("---")

# ---------------- SKILLS DATABASE ----------------
SKILLS_DB = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "statistics"],
    "Web Developer": ["html", "css", "javascript", "react", "node"],
    "Android Developer": ["java", "kotlin", "android", "firebase"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd"],
    "Cyber Security": ["network security", "penetration testing", "encryption"],
    "UI/UX Designer": ["figma", "wireframe", "prototyping", "design"],
    "General Professional": ["communication", "teamwork", "leadership"]
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()

# ---------------- EXTRACT TEXT ----------------
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# ---------------- PREPROCESS ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.replace("nodejs", "node")
    text = text.replace("reactjs", "react")

    words = text.split()
    words = [w for w in words if len(w) > 2]

    return " ".join(words)

# ---------------- UI ----------------
st.title("Resume Screening System")
uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("Could not read resume properly")
        st.stop()

    st.subheader("Resume Preview")
    st.write(resume_text[:500])

    resume_clean = preprocess(resume_text)
    resume_words = set(resume_clean.split())

    if model is None:
        st.error("Model not loaded")
        st.stop()

    # ---------------- EMBEDDING ----------------
    try:
        resume_embedding = model.encode(resume_clean)
    except Exception as e:
        st.error(f"Encoding failed: {e}")
        st.stop()

    # ---------------- SCORING ----------------
    scores = {}

    for role, skills in SKILLS_DB.items():
        job_text = " ".join(skills)
        job_clean = preprocess(job_text)

        job_embedding = model.encode(job_clean)

        score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        scores[role] = float(score)   # 🔥 FIX: convert to float

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ---------------- BEST ROLE ----------------
    best_role = None
    threshold = 0.3

    if sorted_scores and sorted_scores[0][1] >= threshold:
        best_role = sorted_scores[0][0]

    st.header("Best Role")

    if best_role:
        st.success(f"Best match: {best_role}")
    else:
        st.warning("No strong match found")

    # ---------------- TOP 3 ----------------
    st.header("Top 3 Recommended Roles")

    top_3 = sorted_scores[:3]

    for i, (role, score) in enumerate(top_3, 1):
        st.write(f"{i}. {role} ({round(score*100,2)}%)")

    # ---------------- MATCH SCORE ----------------
    if best_role:
        score_percent = sorted_scores[0][1] * 100
        st.header("Resume Score")
        st.metric("Match Score", f"{round(score_percent,2)}%")

    # ---------------- RANKING ----------------
   st.header("Ranking")

for i, (role, score) in enumerate(sorted_scores, 1):

    # 🔥 FIX: Clamp value between 0 and 100
    progress_value = int(max(0, min(score * 100, 100)))

    st.progress(progress_value)
    st.write(f"{i}. {role} ({round(score*100,2)}%)")

    # ---------------- MISSING SKILLS ----------------
    st.header("Missing Skills")

    if best_role and best_role in SKILLS_DB:
        job_skills = SKILLS_DB[best_role]
        missing_skills = []

        for skill in job_skills:
            skill_words = skill.lower().split()

            if not all(word in resume_words for word in skill_words):
                missing_skills.append(skill)

        if missing_skills:
            st.warning(", ".join(missing_skills))
        else:
            st.success("No missing skills 🎉")
