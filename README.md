# RaptorAI — AI-Powered Resume Screening & Job Matching System

**B.Tech CSE (Data Science) Mini Project**

## Overview
RaptorAI is an interactive dual-portal recruitment and career intelligence platform built with Python, Streamlit, and Scikit-learn. It automates multi-resume parsing, TF-IDF semantic similarity scoring, skill coverage evaluation, and personalized skill-gap roadmaps.

## Scoring Formula
`Final Match Score = 0.65 * TF-IDF Cosine Similarity + 0.35 * Required Skill Coverage`

## Project Structure
```text
raptorai_project/
├── .streamlit/
│   └── config.toml          # Light SaaS theme configuration (#F5F7FB + #7657D6)
├── sample_resumes/
│   └── Aarav_Sharma_Sample_Resume.txt
├── pymini.py                # Main Streamlit application
├── requirements.txt         # Python dependencies
├── PROJECT_REPORT.md        # Complete mini-project documentation & Viva Q&A
└── README.md                # Setup and execution guide
```

### 1. Create & Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
*(Note: `nlp_engine.py` includes an automatic fallback parser so RaptorAI still runs cleanly even if `en_core_web_sm` is not downloaded.)*

### 3. Run the Application
```bash
streamlit run pymini.py
