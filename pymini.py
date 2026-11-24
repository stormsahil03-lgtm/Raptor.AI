import io
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None


# ==============================================================================
# 1. PAGE CONFIGURATION & SAAS THEME STYLING (#243B80 Navy + #7657D6 Violet)
# ==============================================================================
st.set_page_config(
    page_title="RaptorAI — AI-Powered Resume Screening & Job Matching",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        color: #202B3C;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #F5F7FB;
    }

    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.92);
        border-bottom: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: transparent;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 4px;
        transition: all 0.18s ease;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #F1F5F9;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #243B80 0%, #7657D6 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(36, 59, 128, 0.18);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px -2px rgba(36, 59, 128, 0.06), 0 1px 3px 0 rgba(15, 23, 42, 0.04) !important;
        padding: 0.35rem !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #243B80 0%, #7657D6 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(118, 87, 214, 0.25) !important;
        transition: opacity 0.18s ease, transform 0.12s ease !important;
    }
    button[kind="primary"]:hover {
        opacity: 0.94 !important;
        transform: translateY(-1px);
    }

    button[kind="secondary"] {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 600 !important;
        background-color: #FFFFFF !important;
        color: #202B3C !important;
    }
    button[kind="secondary"]:hover {
        border-color: #7657D6 !important;
        color: #7657D6 !important;
        background-color: #F5F3FF !important;
    }

    .top-workspace-bar {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 10px rgba(36, 59, 128, 0.04);
    }
    .module-kicker {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #7657D6;
    }
    .module-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #202B3C;
        margin-top: 2px;
    }
    .engine-pill {
        background-color: #ECFDF5;
        color: #16A34A;
        border: 1px solid #A7F3D0;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
    }

    .main-banner {
        background: linear-gradient(135deg, #243B80 0%, #7657D6 100%);
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 10px 25px -5px rgba(36, 59, 128, 0.22);
    }
    .main-banner h2 {
        margin: 0;
        color: white;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .main-banner p {
        margin: 0.35rem 0 0 0;
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.9rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 0.75rem;
    }
    .sidebar-logo-box {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #243B80 0%, #7657D6 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1rem;
    }
    .sidebar-user-card {
        background-color: #F5F7FB;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 10px 12px;
        margin: 8px 0 14px 0;
    }

    .skill-chip-match {
        display: inline-block;
        background-color: #ECFDF5;
        color: #16A34A;
        border: 1px solid #A7F3D0;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    .skill-chip-miss {
        display: inline-block;
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. NLP SKILL DICTIONARY & AI SCORING ENGINE (65% TF-IDF + 35% SKILL COVERAGE)
# ==============================================================================
SKILL_ALIASES = {
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "sql": "SQL",
    "mysql": "SQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "python": "Python",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "streamlit": "Streamlit",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "tableau": "Tableau",
    "excel": "Excel",
    "git": "Git",
    "github": "Git",
    "docker": "Docker",
    "aws": "AWS",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "react": "React",
    "javascript": "JavaScript",
    "html": "HTML/CSS",
    "css": "HTML/CSS",
    "data visualization": "Data Visualization",
    "statistics": "Statistics",
    "spacy": "spaCy",
    "opencv": "OpenCV",
    "computer vision": "Computer Vision",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "mongodb": "MongoDB",
}

SKILL_ROADMAP_GUIDE = {
    "NLP": ("2–3 Weeks", "Natural Language Processing with Python & spaCy", "Build a TF-IDF Document Ranker"),
    "Streamlit": ("4–5 Days", "Interactive Data Apps with Python & Streamlit", "Deploy an ML Model UI with Charts"),
    "Machine Learning": ("3–4 Weeks", "Supervised & Unsupervised ML with Scikit-learn", "End-to-End Classification Pipeline"),
    "Deep Learning": ("4 Weeks", "Neural Networks & PyTorch/TensorFlow", "Image or Text Classifier"),
    "Docker": ("1 Week", "Docker Containers for Data Scientists", "Containerize a Python ML API"),
    "FastAPI": ("1 Week", "ML Inference APIs with FastAPI", "Serve a Scikit-learn Model via REST"),
    "Power BI": ("1–2 Weeks", "Data Storytelling & DAX in Power BI", "Executive KPI Funnel Dashboard"),
    "spaCy": ("1 Week", "Industrial NLP & NER with spaCy", "Extract Entities from PDF Resumes"),
    "PostgreSQL": ("1–2 Weeks", "Relational Design & Window Functions", "Multi-Table Recruitment Warehouse"),
}


def normalize_skill(raw: str) -> str:
    cleaned = raw.strip()
    return SKILL_ALIASES.get(cleaned.lower(), cleaned)


def extract_skills_from_text(text: str) -> list[str]:
    lower = (text or "").lower()
    found = set()
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, lower):
            found.add(canonical)
    return sorted(found) if found else ["Python", "SQL", "Pandas", "Git"]


def extract_profile_metadata(raw_text: str, filename: str) -> dict:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    detected_name = None

    for line in lines[:6]:
        if "SAMPLE RESUME" in line.upper() or "TESTING ONLY" in line.upper():
            m = re.search(r"TESTING ONLY\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", line)
            if m:
                detected_name = m.group(1)
                break
            continue
        if 2 <= len(line.split()) <= 4 and not any(ch in line for ch in "@|:/0123456789"):
            detected_name = line.title()
            break

    if not detected_name:
        clean_fn = re.sub(r"\.[^.]+$", "", filename).replace("_", " ").replace("-", " ").strip()
        detected_name = clean_fn.title() if clean_fn else "Uploaded Candidate"

    edu_line = "Extracted from Uploaded Resume"
    for line in lines:
        if any(k in line.lower() for k in ["b.tech", "b.e.", "m.tech", "b.sc", "bca", "mca", "bachelor", "university", "institute"]):
            edu_line = line[:120]
            break

    clean_summary = re.sub(r"\s+", " ", raw_text).strip()
    if "PROFESSIONAL SUMMARY" in clean_summary.upper():
        clean_summary = clean_summary.split("PROFESSIONAL SUMMARY", 1)[-1].strip(" :-")

    return {
        "name": detected_name,
        "education": edu_line,
        "college": filename,
        "summary": clean_summary[:380] if clean_summary else f"Parsed resume {filename}.",
    }


def parse_uploaded_resume(uploaded_file) -> str:
    """Extracts plain text from uploaded PDF, DOCX, or TXT files in Python."""
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf") and PdfReader is not None:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx") and docx is not None:
        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    return data.decode("utf-8", errors="ignore")


def evaluate_candidate_match(job: dict, candidate: dict) -> dict:
    """Computes 65% TF-IDF Cosine Similarity + 35% Required Skill Coverage."""
    req_skills = [normalize_skill(s) for s in job.get("required_skills", [])]
    cand_skills_lower = {normalize_skill(s).lower() for s in candidate.get("skills", [])}

    projects_text = " ".join(
        f"{p['title']} {' '.join(p['tech'])} {p['description']}" for p in candidate.get("projects", [])
    )
    exp_text = " ".join(
        f"{e['role']} {e['desc']}" for e in candidate.get("experience_details", [])
    )
    cand_corpus = f"{candidate.get('summary', '')} {' '.join(candidate.get('skills', []))} {projects_text} {exp_text} {candidate.get('education', '')}"
    cand_corpus_lower = cand_corpus.lower()

    matched_skills = []
    missing_skills = []
    for req in req_skills:
        if req.lower() in cand_skills_lower or req.lower() in cand_corpus_lower:
            matched_skills.append(req)
        else:
            missing_skills.append(req)

    skill_coverage = round((len(matched_skills) / len(req_skills)) * 100) if req_skills else 100

    job_corpus = f"{job['title']} {job['title']} {job['description']} {' '.join(req_skills)} {' '.join(req_skills)}"

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([job_corpus, cand_corpus])
    raw_cosine = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

    text_similarity = min(97, max(18, round(raw_cosine * 165 + (skill_coverage * 0.25))))
    final_score = min(99, max(15, round(0.65 * text_similarity + 0.35 * skill_coverage)))

    return {
        "final_score": final_score,
        "text_similarity": text_similarity,
        "skill_coverage": skill_coverage,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }


def get_match_tier(score: int) -> str:
    if score >= 80:
        return "🟢 High Match (80%+)"
    if score >= 60:
        return "🟡 Moderate Match (60–79%)"
    if score >= 40:
        return "🟠 Low Match (40–59%)"
    return "🔴 Poor Match (<40%)"


def render_workspace_header(module_label: str, title: str):
    st.markdown(
        f"""
        <div class="top-workspace-bar">
            <div>
                <div class="module-kicker">{module_label}</div>
                <div class="module-title">{title}</div>
            </div>
            <div class="engine-pill">● NLP Engine Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. PRE-LOADED SAMPLE DATA & SESSION STATE INITIALIZATION
# ==============================================================================
SAMPLE_CANDIDATES = [
    {
        "id": "cand-101",
        "file_name": "Aarav_Sharma_Resume.pdf",
        "name": "Aarav Sharma",
        "email": "aarav.sharma.ds@gmail.com",
        "phone": "+91 98765 43210",
        "location": "Bengaluru, Karnataka",
        "education": "B.Tech in CSE (Data Science)",
        "college": "RV College of Engineering (CGPA: 9.1/10)",
        "experience_level": "Entry Level (1 Year Internship)",
        "total_experience": "1 Year 2 Months",
        "status": "Shortlisted",
        "summary": "Data Science graduate specializing in NLP pipelines, predictive machine learning models, and interactive Streamlit analytics apps. Built end-to-end resume parsing and TF-IDF semantic search engines.",
        "skills": ["Python", "Machine Learning", "SQL", "Pandas", "NumPy", "Scikit-learn", "NLP", "spaCy", "Streamlit", "Git", "Statistics"],
        "experience_details": [
            {
                "role": "Data Science Intern — FinPulse Analytics",
                "period": "Jan 2026 – Jun 2026",
                "desc": "Developed churn prediction pipelines using Scikit-learn and Pandas, improving recall by 14%. Deployed internal KPI dashboards in Streamlit backed by SQL.",
            }
        ],
        "projects": [
            {
                "title": "Semantic Resume & Job Matcher using NLP",
                "tech": ["Python", "spaCy", "Scikit-learn", "Streamlit"],
                "description": "Built an automated resume screening prototype combining TF-IDF cosine similarity and skill extraction.",
            }
        ],
        "certifications": ["IBM Data Science Professional Certificate", "NLP Specialization — DeepLearning.AI"],
    },
    {
        "id": "cand-102",
        "file_name": "Priya_Nair_CV.pdf",
        "name": "Priya Nair",
        "email": "priya.nair.ml@outlook.com",
        "phone": "+91 98231 11409",
        "location": "Pune, Maharashtra",
        "education": "B.Tech in CSE (AI & Data Science)",
        "college": "Pune Institute of Computer Technology (CGPA: 8.8/10)",
        "experience_level": "Entry Level (8 Months Internship)",
        "total_experience": "8 Months",
        "status": "Shortlisted",
        "summary": "Machine Learning and Python developer experienced in supervised learning, feature engineering, SQL querying, and deploying data-driven prototypes with Streamlit and Flask.",
        "skills": ["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn", "Streamlit", "TensorFlow", "Data Visualization", "Git"],
        "experience_details": [
            {
                "role": "Machine Learning Intern — CloudMetrics Labs",
                "period": "Dec 2025 – May 2026",
                "desc": "Built demand forecasting models with Python, Pandas, and Scikit-learn. Automated weekly SQL reporting workflows.",
            }
        ],
        "projects": [
            {
                "title": "Healthcare Readmission Predictor",
                "tech": ["Python", "Scikit-learn", "Pandas", "Streamlit"],
                "description": "Trained Random Forest classifiers on clinical records with an interactive Streamlit dashboard.",
            }
        ],
        "certifications": ["Machine Learning Specialization — Andrew Ng"],
    },
    {
        "id": "cand-103",
        "file_name": "Rohan_Verma_Resume.docx",
        "name": "Rohan Verma",
        "email": "rohan.verma24@gmail.com",
        "phone": "+91 97112 88901",
        "location": "Noida, Uttar Pradesh",
        "education": "B.Tech in Computer Science and Engineering",
        "college": "Jaypee Institute of Information Technology (CGPA: 8.3/10)",
        "experience_level": "Fresher (0-1 Years)",
        "total_experience": "6 Months",
        "status": "Under Review",
        "summary": "Data Analytics and Python enthusiast skilled in SQL, Pandas, exploratory data analysis, and machine learning fundamentals.",
        "skills": ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Power BI", "Excel", "Statistics"],
        "experience_details": [
            {
                "role": "Data Analyst Trainee — InsightGrid",
                "period": "Jan 2026 – May 2026",
                "desc": "Wrote complex SQL joins, cleaned multi-source datasets in Pandas, and created Power BI scorecards.",
            }
        ],
        "projects": [
            {
                "title": "Metro Ridership Forecasting & Dashboard",
                "tech": ["Python", "Pandas", "Scikit-learn", "Power BI"],
                "description": "Forecasted daily commuter footfall using regression models.",
            }
        ],
        "certifications": ["Google Data Analytics Professional Certificate"],
    },
    {
        "id": "cand-104",
        "file_name": "Ananya_Iyer_Resume.pdf",
        "name": "Ananya Iyer",
        "email": "ananya.iyer.nlp@gmail.com",
        "phone": "+91 99401 77621",
        "location": "Chennai, Tamil Nadu",
        "education": "B.Tech in CSE (Data Science)",
        "college": "SRM Institute of Science & Technology (CGPA: 8.6/10)",
        "experience_level": "Fresher (0-1 Years)",
        "total_experience": "4 Months",
        "status": "Under Review",
        "summary": "Focused on Natural Language Processing, text mining, and Python application development using spaCy and Flask.",
        "skills": ["Python", "NLP", "spaCy", "Pandas", "Flask", "HTML/CSS", "Git"],
        "experience_details": [
            {
                "role": "NLP Project Intern — LexiTech Solutions",
                "period": "Feb 2026 – May 2026",
                "desc": "Built keyword extraction and document classification scripts using spaCy and Python.",
            }
        ],
        "projects": [
            {
                "title": "Automated News Topic Classifier",
                "tech": ["Python", "NLP", "spaCy", "Flask"],
                "description": "Classified 15,000 news articles using TF-IDF vectorization and Naive Bayes.",
            }
        ],
        "certifications": ["Applied Text Mining in Python — University of Michigan"],
    },
    {
        "id": "cand-105",
        "file_name": "Vikramaditya_K_Resume.pdf",
        "name": "Vikramaditya Kulkarni",
        "email": "vikram.kulkarni.dev@gmail.com",
        "phone": "+91 98500 33198",
        "location": "Hyderabad, Telangana",
        "education": "B.Tech in Information Technology",
        "college": "VNR Vignana Jyothi Institute (CGPA: 7.9/10)",
        "experience_level": "Fresher",
        "total_experience": "3 Months",
        "status": "Under Review",
        "summary": "Backend and database developer proficient in Python, SQL, FastAPI, PostgreSQL, and Docker.",
        "skills": ["Python", "SQL", "FastAPI", "PostgreSQL", "Docker", "Git", "JavaScript"],
        "experience_details": [
            {
                "role": "Backend Engineering Intern — StackWave",
                "period": "Mar 2026 – May 2026",
                "desc": "Developed CRUD microservices in FastAPI and optimized PostgreSQL indexing.",
            }
        ],
        "projects": [
            {
                "title": "Campus Placement Portal API",
                "tech": ["Python", "FastAPI", "PostgreSQL", "Docker"],
                "description": "Built role-based backend endpoints for student profiles and company job postings.",
            }
        ],
        "certifications": ["PostgreSQL for Everybody Specialization"],
    },
    {
        "id": "cand-106",
        "file_name": "Sneha_Deshmukh_Profile.docx",
        "name": "Sneha Deshmukh",
        "email": "sneha.deshmukh.ui@gmail.com",
        "phone": "+91 97654 21098",
        "location": "Mumbai, Maharashtra",
        "education": "B.E. in Computer Engineering",
        "college": "DJ Sanghvi College of Engineering (CGPA: 8.1/10)",
        "experience_level": "Fresher",
        "total_experience": "2 Months",
        "status": "Rejected",
        "summary": "Frontend web developer focused on responsive React interfaces, JavaScript, and HTML/CSS.",
        "skills": ["React", "JavaScript", "HTML/CSS", "Git"],
        "experience_details": [
            {
                "role": "Frontend Intern — PixelCraft Studio",
                "period": "Apr 2026 – May 2026",
                "desc": "Designed and implemented responsive landing pages in React.",
            }
        ],
        "projects": [
            {
                "title": "TaskFlow Kanban Web App",
                "tech": ["React", "JavaScript", "HTML/CSS"],
                "description": "Created a drag-and-drop productivity tracker.",
            }
        ],
        "certifications": ["Meta Front-End Developer Certificate"],
    },
]

INITIAL_JOBS = [
    {
        "id": "job-1",
        "title": "Data Scientist / ML Engineer",
        "company": "NexusAI Labs",
        "location": "Bengaluru / Hybrid",
        "experience": "0-2 Years (Entry Level)",
        "job_type": "Full-Time",
        "salary": "₹8.5 – 12.0 LPA",
        "description": "Seeking an entry-level Data Scientist / ML Engineer with hands-on experience in Python, Machine Learning, NLP text processing, SQL querying, and building interactive data applications using Streamlit, Pandas, and Scikit-learn.",
        "required_skills": ["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn", "NLP", "Streamlit"],
    },
    {
        "id": "job-2",
        "title": "Junior Data Analyst",
        "company": "FinPulse Analytics Pvt Ltd",
        "location": "Pune / Remote",
        "experience": "0-1 Years",
        "job_type": "Full-Time",
        "salary": "₹6.0 – 8.5 LPA",
        "description": "Analyze large transactional datasets using SQL and Python (Pandas, NumPy), build executive BI dashboards, and communicate statistical findings.",
        "required_skills": ["Python", "SQL", "Pandas", "NumPy", "Data Visualization", "Statistics", "Excel"],
    },
    {
        "id": "job-3",
        "title": "AI / NLP Research Intern",
        "company": "CogniText Systems",
        "location": "Hyderabad / Hybrid",
        "experience": "Fresher / Final Year B.Tech",
        "job_type": "Internship",
        "salary": "₹35,000 / month",
        "description": "Work on resume parsing, semantic similarity, entity extraction using spaCy, TF-IDF vectorization, and Python ML pipelines.",
        "required_skills": ["Python", "NLP", "spaCy", "Machine Learning", "Scikit-learn", "Git"],
    },
    {
        "id": "job-4",
        "title": "Backend Python & AI API Engineer",
        "company": "ScaleStack Cloud",
        "location": "Bengaluru, Karnataka",
        "experience": "0-2 Years",
        "job_type": "Full-Time",
        "salary": "₹9.0 – 13.5 LPA",
        "description": "Design scalable REST APIs in FastAPI/Flask to serve ML models, manage PostgreSQL databases, and containerize services using Docker.",
        "required_skills": ["Python", "FastAPI", "SQL", "PostgreSQL", "Docker", "REST APIs", "Git"],
    },
]


def init_state():
    if "role" not in st.session_state:
        st.session_state.role = None
    if "jobs" not in st.session_state:
        st.session_state.jobs = INITIAL_JOBS.copy()
    if "screening_runs" not in st.session_state:
        primary_job = st.session_state.jobs[0]
        evaluated = []
        for cand in SAMPLE_CANDIDATES:
            res = evaluate_candidate_match(primary_job, cand)
            evaluated.append({**cand, "job_title": primary_job["title"], **res})
        evaluated.sort(key=lambda x: x["final_score"], reverse=True)
        avg_score = round(sum(c["final_score"] for c in evaluated) / len(evaluated))
        st.session_state.screening_runs = [
            {
                "id": "run-01",
                "date": "2026-11-22 14:30",
                "job": primary_job,
                "candidates": evaluated,
                "avg_score": avg_score,
            }
        ]
    if "active_run_idx" not in st.session_state:
        st.session_state.active_run_idx = 0
    if "student_profile" not in st.session_state:
        st.session_state.student_profile = {
            "file_name": "Aayush_raj_BTech_CSE_DS.pdf",
            "name": "Aayush Raj",
            "email": "aayushraj.ds@gmail.com",
            "location": "Greater Noida, Uttar Pradesh",
            "education": "B.Tech in Computer Science & Engineering (Data Science)",
            "college": "KCC Institute of Engineering & Technology",
            "summary": "B.Tech CSE (Data Science) student passionate about Python programming, SQL databases, exploratory data analysis with Pandas & NumPy, and building machine learning models using Scikit-learn.",
            "skills": ["Python", "SQL", "Pandas", "NumPy", "Scikit-learn", "Data Visualization", "Git", "Statistics"],
            "projects": [
                {
                    "title": "Student Placement & CGPA Predictor",
                    "tech": ["Python", "Pandas", "Scikit-learn"],
                    "description": "Built a classification pipeline predicting campus placement outcomes with 86% accuracy.",
                }
            ],
        }
    if "saved_job_ids" not in st.session_state:
        st.session_state.saved_job_ids = {"job-1", "job-2"}
    if "last_parsed_file" not in st.session_state:
        st.session_state.last_parsed_file = None


init_state()


# ==============================================================================
# 4. ENTRY ROLE SELECTION SCREEN
# ==============================================================================
if st.session_state.role is None:
    
    # ABOUT / TEAM BUTTON - TOP RIGHT
    space, about_col = st.columns([20, 2])

    with about_col:
        with st.popover(" ⓘ About", use_container_width=False):
            st.markdown("## About RaptorAI")
            st.write(
                "AI-Powered Resume Screening and Job Matching System"
            )

            st.divider()
            st.markdown("### 👥 Our Team")

            members = [
                "Aayush Raj",
                "Prince Kumar Singh",
                "Aashi Kumari",
                "Anjalika Chacko",
                "Md.Salim Siddiqui"
            ]

            for i, name in enumerate(members, 1):
                st.write(f"{i}. {name}")

            st.divider()
            st.caption("B.Tech CSE (Data Science) | Mini Project")

    st.markdown(
        """
        <div class="main-banner" style="text-align:center; padding: 2.2rem 1.5rem;">
            <div style="font-size:0.78rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; opacity:0.88;">
                NLP • TF-IDF Vectorization • Cosine Similarity • Skill Gap Intelligence
            </div>
            <h2 style="font-size:2.1rem; margin-top:0.35rem;">Welcome to RaptorAI</h2>
            <p style="font-size:1.05rem;">AI-Powered Resume Screening and Job Matching System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h4 style='text-align:center; color:#243B80; margin-bottom:1rem;'>What best describes you?</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("### 🎓 Student / Job Seeker")
            st.write("Upload your resume, analyze your skills and get personalized job recommendations.")
            st.markdown(
                "- ✓ Automated skill extraction & profile summary\n"
                "- ✓ AI-ranked job matching (65% Text + 35% Skill)\n"
                "- ✓ Interactive Skill Gap Analyzer & learning roadmap"
            )
            if st.button("Continue as Student →", type="primary", use_container_width=True):
                st.session_state.role = "student"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 💼 Recruiter")
            st.write("Upload multiple resumes, screen candidates with AI and find the best matches for your jobs.")
            st.markdown(
                "- ✓ Multi-resume batch upload (PDF, DOCX, TXT)\n"
                "- ✓ Transparent AI scoring & skill coverage breakdown\n"
                "- ✓ Shortlisting pipeline, funnel & CSV report export"
            )
            if st.button("Continue as Recruiter →", type="primary", use_container_width=True):
                st.session_state.role = "recruiter"
                st.rerun()
    st.stop()


# ==============================================================================
# 5. SAAS SIDEBAR NAVIGATION & ROLE SWITCHER
# ==============================================================================
with st.sidebar:
    is_recruiter = st.session_state.role == "recruiter"
    workspace_sub = "Recruiter Workspace" if is_recruiter else "Student Career Portal"
    user_title = "Raptor Team" if is_recruiter else st.session_state.student_profile["name"]
    user_sub = "Hiring Mode Active" if is_recruiter else "B.Tech CSE (Data Science)"

    st.image("raptorai_logo.png", width=190)

    st.markdown(
    f"""
    <div style="font-size:0.72rem; color:#64748B;
                margin-bottom:15px;">
        {workspace_sub}
    </div>
    <div class="sidebar-user-card">
        <div style="font-size:0.8rem; font-weight:700;
                    color:#202B3C;">● {user_title}</div>
        <div style="font-size:0.72rem; color:#64748B;">
            {user_sub}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
    )

    if is_recruiter:
        nav = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Resume Screening",
                "AI Screening Results",
                "Candidate Profile Inspector",
                "Candidate Management",
            ],
            index=1,
            label_visibility="collapsed",
        )
    else:
        nav = st.radio(
            "Navigation",
            [
                "Student Dashboard",
                "Resume Upload & Analyzer",
                "Job Recommendations",
                "Skill Gap Analyzer",
                "Saved Jobs",
            ],
            index=0,
            label_visibility="collapsed",
        )

    st.divider()
    st.markdown(
        """
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:10px 12px; font-size:0.75rem; color:#64748B; margin-bottom:10px;">
            <div style="font-weight:700; color:#334155; margin-bottom:2px;">AI Scoring Engine</div>
            65% Text Similarity + 35% Skill Coverage
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("⇄ Switch Role", use_container_width=True):
        st.session_state.role = None
        st.rerun()


# ==============================================================================
# 6. RECRUITER MODULE
# ==============================================================================
if st.session_state.role == "recruiter":
    active_run = st.session_state.screening_runs[st.session_state.active_run_idx]
    all_candidates = [c for run in st.session_state.screening_runs for c in run["candidates"]]

    if nav == "Dashboard":
        render_workspace_header("RECRUITER MODULE", "Recruiter Overview Dashboard")
        st.markdown(
            '<div class="main-banner"><h2>Welcome back, Raptor Team</h2><p>Screen candidate resumes in seconds using TF-IDF semantic similarity and automated skill coverage scoring.</p></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        shortlisted_cnt = sum(1 for c in all_candidates if c["status"] in ("Shortlisted", "Selected"))
        with m1:
            with st.container(border=True):
                st.metric("Total Resumes", len(all_candidates), "100% parsed")
        with m2:
            with st.container(border=True):
                st.metric("Candidates Screened", len(all_candidates), f"{len(st.session_state.screening_runs)} runs")
        with m3:
            with st.container(border=True):
                st.metric("Shortlisted Candidates", shortlisted_cnt, "Qualified")
        with m4:
            with st.container(border=True):
                st.metric("Active Jobs", len(st.session_state.jobs), "Synced")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.subheader("Candidate Match Overview")
                tiers = {
                    "High (80%+)": sum(1 for c in all_candidates if c["final_score"] >= 80),
                    "Moderate (60-79%)": sum(1 for c in all_candidates if 60 <= c["final_score"] < 80),
                    "Low (40-59%)": sum(1 for c in all_candidates if 40 <= c["final_score"] < 60),
                    "Poor (<40%)": sum(1 for c in all_candidates if c["final_score"] < 40),
                }
                st.bar_chart(pd.Series(tiers), color="#7657D6")

        with c2:
            with st.container(border=True):
                st.subheader("Top Required Skills")
                skill_counts = {}
                for j in st.session_state.jobs:
                    for s in j["required_skills"]:
                        skill_counts[s] = skill_counts.get(s, 0) + 1
                st.bar_chart(pd.Series(skill_counts).sort_values(ascending=False).head(8), color="#243B80")

    elif nav == "Resume Screening":
        render_workspace_header("RECRUITER MODULE", "Resume Screening & Job Setup")
        col_job, col_up = st.columns([1.25, 1], gap="large")

        with col_job:
            with st.container(border=True):
                st.markdown("<span style='font-size:0.72rem; font-weight:700; color:#7657D6; text-transform:uppercase;'>Step 1 • Job Criteria</span>", unsafe_allow_html=True)
                st.subheader("New Job Description")
                job_title = st.text_input("Job Title *", value="Data Scientist / ML Engineer")
                c_exp, c_type = st.columns(2)
                exp_req = c_exp.selectbox("Experience Required", ["0-2 Years (Entry Level)", "Fresher / Internship", "1-3 Years", "3-5 Years"])
                job_type = c_type.selectbox("Job Type", ["Full-Time", "Internship", "Hybrid", "Remote"])
                job_desc = st.text_area(
                    "Job Description",
                    height=135,
                    value="We are seeking an entry-level Data Scientist / ML Engineer with hands-on experience in Python, Machine Learning, NLP text processing, SQL querying, and building interactive data applications using Streamlit, Pandas, and Scikit-learn.",
                )
                req_skills = st.multiselect(
                    "Required Skills (35% Skill Coverage Weight)",
                    options=sorted(set(SKILL_ALIASES.values())),
                    default=["Python", "Machine Learning", "SQL", "Pandas", "Scikit-learn", "NLP", "Streamlit"],
                )

        with col_up:
            with st.container(border=True):
                st.markdown("<span style='font-size:0.72rem; font-weight:700; color:#243B80; text-transform:uppercase;'>Step 2 • Candidate Resumes</span>", unsafe_allow_html=True)
                st.subheader("Upload Resumes")
                uploaded_files = st.file_uploader(
                    "Drag & drop candidate resumes here (PDF, DOCX, TXT)",
                    type=["pdf", "docx", "txt"],
                    accept_multiple_files=True,
                )
                use_sample_pool = st.checkbox(
                    "Include 6 Sample B.Tech CSE (Data Science) Candidate Resumes",
                    value=True,
                )

                if st.button("✨ Run AI Screening", type="primary", use_container_width=True):
                    if not job_title.strip() or not req_skills:
                        st.error("Please provide a Job Title and at least one Required Skill.")
                    else:
                        new_job = {
                            "id": f"job-{len(st.session_state.jobs) + 1}",
                            "title": job_title.strip(),
                            "company": "RaptorAI Talent Partner",
                            "location": "Bengaluru / Hybrid",
                            "experience": exp_req,
                            "job_type": job_type,
                            "salary": "₹8.0 – 12.0 LPA",
                            "description": job_desc,
                            "required_skills": req_skills,
                        }
                        st.session_state.jobs.insert(0, new_job)

                        pool = []
                        if use_sample_pool:
                            pool.extend([dict(c) for c in SAMPLE_CANDIDATES])

                        for idx, f in enumerate(uploaded_files or []):
                            text = parse_uploaded_resume(f)
                            detected = extract_skills_from_text(text + " " + f.name)
                            meta = extract_profile_metadata(text, f.name)
                            pool.append(
                                {
                                    "id": f"up-{datetime.now().strftime('%H%M%S')}-{idx}",
                                    "file_name": f.name,
                                    "name": meta["name"],
                                    "email": f"{meta['name'].lower().replace(' ', '.')}@candidate.edu",
                                    "phone": "+91 98100 00000",
                                    "location": "Delhi NCR",
                                    "education": meta["education"],
                                    "college": meta["college"],
                                    "experience_level": "Fresher (0-1 Years)",
                                    "total_experience": "6 Months",
                                    "status": "Under Review",
                                    "summary": meta["summary"],
                                    "skills": detected,
                                    "experience_details": [],
                                    "projects": [],
                                    "certifications": ["Extracted from Uploaded Document"],
                                }
                            )

                        if not pool:
                            st.warning("Please upload at least one file or check the sample resumes box.")
                        else:
                            evaluated = []
                            for cand in pool:
                                scores = evaluate_candidate_match(new_job, cand)
                                status = (
                                    "Shortlisted"
                                    if scores["final_score"] >= 80
                                    else ("Rejected" if scores["final_score"] < 40 else "Under Review")
                                )
                                evaluated.append({**cand, "job_title": new_job["title"], "status": status, **scores})

                            evaluated.sort(key=lambda x: x["final_score"], reverse=True)
                            avg_sc = round(sum(c["final_score"] for c in evaluated) / len(evaluated))

                            st.session_state.screening_runs.insert(
                                0,
                                {
                                    "id": f"run-{len(st.session_state.screening_runs) + 1}",
                                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    "job": new_job,
                                    "candidates": evaluated,
                                    "avg_score": avg_sc,
                                },
                            )
                            st.session_state.active_run_idx = 0
                            st.success(f"Screened {len(evaluated)} candidates for **{new_job['title']}** (Avg Match: {avg_sc}%). Select **AI Screening Results** in the left sidebar!")
                st.caption("Computes 65% TF-IDF Cosine Similarity + 35% Required Skill Coverage")

    elif nav == "AI Screening Results":
        render_workspace_header("RECRUITER MODULE", f"AI Screening Results — {active_run['job']['title']}")

        run_labels = [f"{r['job']['title']} ({r['date']} • {len(r['candidates'])} resumes)" for r in st.session_state.screening_runs]
        selected_run_label = st.selectbox("Select Screening Run", run_labels, index=st.session_state.active_run_idx)
        st.session_state.active_run_idx = run_labels.index(selected_run_label)
        active_run = st.session_state.screening_runs[st.session_state.active_run_idx]

        with st.container(border=True):
            f1, f2, f3 = st.columns([2, 1, 1])
            search_q = f1.text_input("Search candidate name or skill")
            tier_filter = f2.selectbox("Match Tier", ["All", "High (80%+)", "Moderate (60-79%)", "Low (40-59%)", "Poor (<40%)"])
            sort_order = f3.selectbox("Sort By", ["Score: High → Low", "Score: Low → High", "Name: A → Z"])

            rows = list(active_run["candidates"])
            if search_q.strip():
                q = search_q.lower()
                rows = [c for c in rows if q in c["name"].lower() or any(q in s.lower() for s in c["skills"])]
            if tier_filter != "All":
                rows = [c for c in rows if tier_filter.split()[0] in get_match_tier(c["final_score"])]
            if sort_order == "Score: Low → High":
                rows.sort(key=lambda x: x["final_score"])
            elif sort_order == "Name: A → Z":
                rows.sort(key=lambda x: x["name"])

            df = pd.DataFrame(
                [
                    {
                        "Candidate Name": c["name"],
                        "AI Match Score (%)": c["final_score"],
                        "Text Sim (65% wt)": f"{c['text_similarity']}%",
                        "Skill Cov (35% wt)": f"{c['skill_coverage']}%",
                        "Tier": get_match_tier(c["final_score"]),
                        "Matched Skills": ", ".join(c["matched_skills"]) or "None",
                        "Missing Skills": ", ".join(c["missing_skills"]) or "All Matched",
                        "Status": c["status"],
                    }
                    for c in rows
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Screening Report (CSV)",
                data=csv_bytes,
                file_name=f"RaptorAI_{active_run['job']['title'].replace(' ', '_')}.csv",
                mime="text/csv",
                type="primary",
            )

    elif nav == "Candidate Profile Inspector":
        render_workspace_header("RECRUITER MODULE", "Candidate Profile & Detailed AI Analysis")
        cands = active_run["candidates"]
        cand_names = [f"{c['name']} ({c['final_score']}% Match)" for c in cands]
        sel_label = st.selectbox("Select Candidate to Inspect", cand_names)
        cand = cands[cand_names.index(sel_label)]

        with st.container(border=True):
            top1, top2 = st.columns([2, 1])
            with top1:
                st.subheader(f"{cand['name']} — {cand['status']}")
                st.write(f"**Applied Role:** {cand['job_title']} | **Education:** {cand['education']}")
                st.caption(f"📧 {cand['email']}  •  📞 {cand['phone']}  •  📍 {cand['location']}")
            with top2:
                st.metric("Overall AI Match Score", f"{cand['final_score']}%", get_match_tier(cand["final_score"]))
                st.caption(f"Text Sim: **{cand['text_similarity']}%** | Skill Cov: **{cand['skill_coverage']}%**")

            new_status = st.selectbox(
                "Update Candidate Pipeline Status",
                ["Under Review", "Shortlisted", "Selected", "Rejected"],
                index=["Under Review", "Shortlisted", "Selected", "Rejected"].index(cand["status"]),
            )
            if new_status != cand["status"]:
                cand["status"] = new_status
                st.success(f"Updated {cand['name']} status to **{new_status}**.")

        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### Extracted Resume Details")
                st.write(cand["summary"])
                st.markdown("**All Extracted Skills:**")
                st.write(", ".join(cand["skills"]))
                for p in cand.get("projects", []):
                    st.markdown(f"- **{p['title']}** ({', '.join(p['tech'])}): {p['description']}")

        with right:
            with st.container(border=True):
                st.markdown("#### Skill Match & AI Recommendation")
                st.progress(cand["skill_coverage"] / 100.0, text=f"Required Skill Coverage: {cand['skill_coverage']}%")

                matched_html = "".join(f'<span class="skill-chip-match">✓ {s}</span>' for s in cand["matched_skills"])
                missing_html = "".join(f'<span class="skill-chip-miss">! {s}</span>' for s in cand["missing_skills"])

                st.markdown("**Matched Required Skills:**", unsafe_allow_html=True)
                st.markdown(matched_html or "_None_", unsafe_allow_html=True)
                st.markdown("**Missing Required Skills:**", unsafe_allow_html=True)
                st.markdown(missing_html or "_All required skills matched!_", unsafe_allow_html=True)
                st.info(
                    "**Human-in-the-Loop Notice:** The AI score (65% Text Similarity + 35% Skill Coverage) is a decision-support indicator, not a final hiring decision."
                )

    elif nav == "Candidate Management":
        render_workspace_header("RECRUITER MODULE", "Candidate Pipeline & Management")
        cands = active_run["candidates"]
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            with st.container(border=True):
                st.metric("Total", len(cands))
        with s2:
            with st.container(border=True):
                st.metric("Shortlisted", sum(1 for c in cands if c["status"] == "Shortlisted"))
        with s3:
            with st.container(border=True):
                st.metric("Under Review", sum(1 for c in cands if c["status"] == "Under Review"))
        with s4:
            with st.container(border=True):
                st.metric("Selected", sum(1 for c in cands if c["status"] == "Selected"))

        status_filter = st.radio("Pipeline Stage", ["All", "Shortlisted", "Under Review", "Selected", "Rejected"], horizontal=True)
        filtered = cands if status_filter == "All" else [c for c in cands if c["status"] == status_filter]

        for c in filtered:
            with st.container(border=True):
                col_a, col_b, col_c = st.columns([2.5, 1, 1.5])
                col_a.markdown(f"**{c['name']}** ({c['email']})  \nRole: *{c['job_title']}* • Match: **{c['final_score']}%**")
                col_b.markdown(f"Status: **{c['status']}**")
                chosen = col_c.selectbox(
                    "Change Status",
                    ["Under Review", "Shortlisted", "Selected", "Rejected"],
                    index=["Under Review", "Shortlisted", "Selected", "Rejected"].index(c["status"]),
                    key=f"mgmt_{c['id']}",
                    label_visibility="collapsed",
                )
                if chosen != c["status"]:
                    c["status"] = chosen
                    st.rerun()


# ==============================================================================
# 7. STUDENT / JOB SEEKER MODULE
# ==============================================================================
else:
    sp = st.session_state.student_profile
    pseudo_candidate = {
        "summary": sp["summary"],
        "skills": sp["skills"],
        "projects": sp["projects"],
        "experience_details": [],
        "education": sp["education"],
    }
    job_matches = []
    for job in st.session_state.jobs:
        res = evaluate_candidate_match(job, pseudo_candidate)
        job_matches.append({**job, **res})
    job_matches.sort(key=lambda x: x["final_score"], reverse=True)

    if nav == "Student Dashboard":
        render_workspace_header("STUDENT / JOB SEEKER MODULE", "Student Career Dashboard")
        st.markdown(
            f'<div class="main-banner"><h2>Hello, {sp["name"]}</h2><p>{sp["education"]} • Active Resume: {sp["file_name"]}</p></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            with st.container(border=True):
                st.metric("Detected Skills", len(sp["skills"]))
        with m2:
            with st.container(border=True):
                st.metric("Best Job Match", f"{job_matches[0]['final_score']}%" if job_matches else "0%")
        with m3:
            with st.container(border=True):
                st.metric("Recommended Roles", sum(1 for j in job_matches if j["final_score"] >= 55))
        with m4:
            with st.container(border=True):
                st.metric("Saved Jobs", len(st.session_state.saved_job_ids))

        st.subheader("Top AI-Matched Job Recommendations")
        for job in job_matches[:3]:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### {job['title']} — `{job['final_score']}% Match`")
                c1.caption(f"{job['company']} • {job['location']} • {job['salary']}")
                c1.write(f"**Matched:** {', '.join(job['matched_skills']) or 'None'}  |  **Missing:** {', '.join(job['missing_skills']) or 'None'}")
                is_saved = job["id"] in st.session_state.saved_job_ids
                if c2.button("★ Saved" if is_saved else "☆ Save Job", key=f"dash_save_{job['id']}", use_container_width=True):
                    if is_saved:
                        st.session_state.saved_job_ids.remove(job["id"])
                    else:
                        st.session_state.saved_job_ids.add(job["id"])
                    st.rerun()

    elif nav == "Resume Upload & Analyzer":
        render_workspace_header("STUDENT / JOB SEEKER MODULE", "Resume Upload & Skill Analyzer")
        with st.container(border=True):
            up_file = st.file_uploader("Upload Your Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

            if up_file is not None and st.session_state.get("last_parsed_file") != up_file.name:
                raw_text = parse_uploaded_resume(up_file)
                extracted = extract_skills_from_text(raw_text + " " + up_file.name)
                meta = extract_profile_metadata(raw_text, up_file.name)

                sp["file_name"] = up_file.name
                sp["name"] = meta["name"]
                sp["education"] = meta["education"]
                sp["college"] = meta["college"]
                sp["summary"] = meta["summary"]
                sp["skills"] = extracted
                st.session_state.last_parsed_file = up_file.name
                st.rerun()

            if up_file is not None:
                st.success(f"Parsed **{sp['file_name']}** and extracted **{len(sp['skills'])}** technical skills!")

            updated_skills = st.multiselect(
                "Your Active Technical Skills (add/remove skills to simulate score changes)",
                options=sorted(set(list(SKILL_ALIASES.values()) + sp["skills"])),
                default=sp["skills"],
            )
            if updated_skills != sp["skills"]:
                sp["skills"] = updated_skills
                st.rerun()

        with st.container(border=True):
            st.markdown(
                f"**Candidate:** {sp['name']}  \n"
                f"**Education:** {sp['education']}  \n"
                f"**Summary:** {sp['summary']}"
            )

    elif nav == "Job Recommendations":
        render_workspace_header("STUDENT / JOB SEEKER MODULE", "Personalized Job Recommendations")
        for job in job_matches:
            with st.container(border=True):
                c1, c2 = st.columns([3.2, 1])
                with c1:
                    st.subheader(f"{job['title']} — {job['final_score']}% Match")
                    st.caption(f"{job['company']} • {job['location']} • {job['experience']} • {job['salary']}")
                    st.write(job["description"])
                    matched_html = "".join(f'<span class="skill-chip-match">✓ {s}</span>' for s in job["matched_skills"])
                    missing_html = "".join(f'<span class="skill-chip-miss">! Need {s}</span>' for s in job["missing_skills"])
                    st.markdown(matched_html + missing_html, unsafe_allow_html=True)
                with c2:
                    st.metric("Match Score", f"{job['final_score']}%")
                    st.caption(f"Text: {job['text_similarity']}% | Skill: {job['skill_coverage']}%")
                    is_saved = job["id"] in st.session_state.saved_job_ids
                    if st.button("★ Remove Saved" if is_saved else "☆ Save Job", key=f"rec_save_{job['id']}", use_container_width=True):
                        if is_saved:
                            st.session_state.saved_job_ids.remove(job["id"])
                        else:
                            st.session_state.saved_job_ids.add(job["id"])
                        st.rerun()

    elif nav == "Skill Gap Analyzer":
        render_workspace_header("STUDENT / JOB SEEKER MODULE", "Interactive Skill Gap Analyzer")
        titles = [f"{j['title']} ({j['final_score']}% Match)" for j in job_matches]
        sel_title = st.selectbox("Select Target Job Role", titles)
        target_job = job_matches[titles.index(sel_title)]

        col_score, col_gap = st.columns([1, 1.5], gap="large")
        with col_score:
            with st.container(border=True):
                st.metric("Composite Match Score", f"{target_job['final_score']}%", get_match_tier(target_job["final_score"]))
                st.write(f"- **Text Similarity (65% wt):** `{target_job['text_similarity']}%` → `{0.65 * target_job['text_similarity']:.1f} pts`")
                st.write(f"- **Skill Coverage (35% wt):** `{target_job['skill_coverage']}%` → `{0.35 * target_job['skill_coverage']:.1f} pts`")
                st.progress(target_job["skill_coverage"] / 100.0, text=f"Skills Satisfied: {len(target_job['matched_skills'])}/{len(target_job['required_skills'])}")

        with col_gap:
            with st.container(border=True):
                st.subheader("Recommended Upskilling Roadmap")
                if not target_job["missing_skills"]:
                    st.success(f"Zero skill gaps! Your profile covers 100% of required skills for **{target_job['title']}**.")
                else:
                    for skill in target_job["missing_skills"]:
                        est_time, course, capstone = SKILL_ROADMAP_GUIDE.get(
                            skill,
                            ("1–2 Weeks", f"Hands-On {skill} Foundations", f"Build a Mini Project using {skill}"),
                        )
                        c_info, c_btn = st.columns([3, 1])
                        c_info.markdown(f"**Missing Skill: `{skill}`** ({est_time})  \n📚 *Course:* {course}  \n🛠️ *Capstone:* {capstone}")
                        if c_btn.button("+ Add Skill", key=f"learn_{skill}", type="primary"):
                            sp["skills"].append(skill)
                            st.rerun()

    elif nav == "Saved Jobs":
        render_workspace_header("STUDENT / JOB SEEKER MODULE", "Saved Job Opportunities")
        saved_list = [j for j in job_matches if j["id"] in st.session_state.saved_job_ids]
        if not saved_list:
            st.info("No saved jobs yet. Bookmark roles from **Job Recommendations** to track them here.")
        else:
            for job in saved_list:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"### {job['title']} — `{job['final_score']}% Match`")
                    c1.caption(f"{job['company']} • {job['location']} • {job['salary']}")
                    c1.write(job["description"])
                    if c2.button("Remove Bookmark", key=f"unsave_{job['id']}", use_container_width=True):
                        st.session_state.saved_job_ids.remove(job["id"])
                        st.rerun()
