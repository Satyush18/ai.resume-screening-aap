AI Resume Analyzer

An AI-powered resume analysis system that uses Natural Language Processing (NLP) and sentence embeddings to evaluate resumes, recommend suitable job roles, and identify missing skills.

Live Demo

https://ai-resume-analyzer-7uwmjhprkmhtwn96skzv27.streamlit.app/

 Model Used

Sentence Transformers ("all-MiniLM-L6-v2")

 Features

 Upload resume in PDF format
 Smart skill detection with normalization (OOP, ML, DP, etc.)
 AI-based job role prediction
Role ranking using similarity scores
 Missing skills identification

Tech Stack

Layer| Technology
Frontend| Streamlit
Backend| Python
NLP Model| Sentence Transformers
Similarity| Scikit-learn
PDF Parsing| PyPDF2

 Workflow

1. Resume Upload
2. Text Extraction from PDF
3. Text Preprocessing & Cleaning
4. Skill Normalization (OOP, ML, etc.)
5. Embedding Generation using NLP model
6. Cosine Similarity Matching
7. Role Prediction & Ranking
8. Missing Skills Detection

 How to Run Locally

pip install -r requirements.txt
streamlit run app.py

Project Structure

AI-Resume-Analyzer

1. app.py
2. requirements.txt
3. README.md

 Screenshots

 Upload Interface

![Upload](assets/upload.png)

 Resume Preview

![Preview](assets/preview.png)

 Best Role Prediction

![Result](assets/result.png)

 Missing Skills Detection

![Missing Skills](assets/missing_skills.png)


- PDF parsing may fail for complex or scanned resumes
- No job description matching
- Static skill database
- Generic roles may rank higher due to common keywords

Future Improvements

- Add job description matching
- Improve parsing using PyMuPDF
- Section-based scoring (Skills > Experience > Projects)
- Build full-stack version (React + Flask)

 Author

Satyush Mohapatra

 Project Type
 
AI-based Resume Analyzer using NLP (not a traditional ATS)
