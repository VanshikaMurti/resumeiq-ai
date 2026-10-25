
import re
import sqlite3
from io import BytesIO
from datetime import datetime
import pytesseract
import fitz
from PIL import Image


import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sentence_transformers import SentenceTransformer, util

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ResumeIQ AI",
    page_icon="💎",
    layout="wide"
)

DB_NAME = "resume_history.db"

# ---------------- DATABASE ----------------
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TEXT NOT NULL,
                role TEXT,
                filename TEXT,
                score REAL,
                matched TEXT,
                missing TEXT
            )
        """)

        # Add filename column if an older database exists
        columns = [
            row[1] for row in
            conn.execute("PRAGMA table_info(analyses)").fetchall()
        ]
        if "filename" not in columns:
            conn.execute(
                "ALTER TABLE analyses ADD COLUMN filename TEXT"
            )


def save_analysis(role, filename, score, matched, missing):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO analyses
            (analyzed_at, role, filename, score, matched, missing)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%d %b %Y, %I:%M %p"),
            role.strip() or "Not specified",
            filename,
            score,
            ", ".join(matched),
            ", ".join(missing)
        ))


def get_history():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT id, analyzed_at, role, filename,
                   score, matched, missing
            FROM analyses
            ORDER BY id DESC
        """).fetchall()


def get_analysis(analysis_id):
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("""
            SELECT id, analyzed_at, role, filename,
                   score, matched, missing
            FROM analyses WHERE id = ?
        """, (analysis_id,)).fetchone()


init_db()

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #080d1b;
    color: #f1f5f9;
    font-family: Inter, sans-serif;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #10182b;
    border-right: 1px solid #263450;
}
.hero {
    padding: 36px;
    border-radius: 24px;
    margin-bottom: 28px;
    background: linear-gradient(120deg,#172554,#4c1d95,#0e7490);
    border: 1px solid #5145a5;
}
.hero h1 {
    font-size: clamp(32px, 4vw, 46px);
    color: white;
    margin: 12px 0;
}
.hero p {
    color: #dbeafe;
    font-size: 16px;
    line-height: 1.8;
}
.metric-card {
    background: linear-gradient(145deg,#151f35,#101729);
    border: 1px solid #344365;
    padding: 22px;
    border-radius: 17px;
    min-height: 120px;
}
.metric-label {
    color: #a5b4d0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}
.metric-value {
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin-top: 12px;
}
.pill {
    display: inline-block;
    margin: 4px;
    padding: 7px 12px;
    border-radius: 30px;
    background: #153a37;
    border: 1px solid #247767;
    color: #86efac;
}
.gap-pill {
    display: inline-block;
    margin: 4px;
    padding: 7px 12px;
    border-radius: 30px;
    background: #3a2040;
    border: 1px solid #79428a;
    color: #f0abfc;
}
.panel {
    background: #111a2d;
    border: 1px solid #293957;
    border-radius: 16px;
    padding: 20px;
}
div.stButton > button,
div.stDownloadButton > button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
    min-height: 44px;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: linear-gradient(90deg,#4f46e5,#7c3aed);
    color: white;
    border: none;
}
.stTextArea textarea {
    background: #111a2d;
    color: white;
    border-radius: 12px;
    border: 1px solid #344365;
}
[data-testid="stFileUploader"] {
    background: #111a2d;
    border: 1px dashed #6366f1;
    border-radius: 13px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SKILLS ----------------
SKILLS = [
    "python", "c++", "java", "sql", "html", "css",
    "javascript", "react", "node.js", "flask",
    "streamlit", "machine learning", "deep learning",
    "data analysis", "pandas", "numpy", "scikit-learn",
    "git", "github", "rest api", "aws", "docker",
    "data structures", "algorithms", "communication",
    "problem solving", "excel", "power bi", "tensorflow",
    "mongodb", "mysql", "postgresql", "fastapi", "oop",
    "operating systems", "dbms", "api", "linux",
    "computer networks", "object oriented programming"
]



def extract_text(pdf_file):
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)

    extracted_pages = [
        page.extract_text() or ""
        for page in reader.pages
    ]

    if all(text.strip() for text in extracted_pages):
        return "\n".join(extracted_pages)

    pdf_bytes = pdf_file.getvalue()
    ocr_pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        for index, page in enumerate(document):
            text = extracted_pages[index]

            if text.strip():
                ocr_pages.append(text)
            else:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )
                image = Image.open(BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(image)
                ocr_pages.append(text)

    return "\n".join(ocr_pages)


def find_skills(text):
    text = text.lower()
    found = []

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return found


# Model downloads on first use and is cached afterwards.
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_score(resume, job):
    model = load_model()

    embeddings = model.encode(
        [resume, job],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    similarity = util.cos_sim(
        embeddings[0], embeddings[1]
    ).item()

    return round(max(0, min(100, similarity * 100)), 1)


def create_pdf_report(result):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    width, height = letter
    y = height - 55

    pdf.setTitle("ResumeIQ Analysis Report")
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(45, y, "ResumeIQ AI - Analysis Report")
    y -= 35

    pdf.setFont("Helvetica", 11)

    details = [
        f"Analysis date: {result['date']}",
        f"Target role: {result['role']}",
        f"Resume file: {result['filename']}",
        f"Semantic similarity: {result['score']}%"
    ]

    for detail in details:
        pdf.drawString(45, y, detail[:110])
        y -= 20

    y -= 10

    sections = [
        ("Matched skills", result["matched"]),
        ("Skills to review", result["missing"])
    ]

    for title, skills in sections:
        if y < 90:
            pdf.showPage()
            y = height - 55

        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(45, y, title)
        y -= 23
        pdf.setFont("Helvetica", 11)

        for skill in skills or ["None detected"]:
            if y < 60:
                pdf.showPage()
                y = height - 55
                pdf.setFont("Helvetica", 11)

            pdf.drawString(55, y, "- " + skill[:100])
            y -= 17

        y -= 12

    if y < 55:
        pdf.showPage()
        y = height - 55

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        45, y,
        "Similarity is an estimate, not an official ATS score or hiring probability."
    )

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def show_metric(label, value):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def show_result(result, historical=False):
    st.markdown("---")
    st.markdown("## 📊 Analysis Dashboard")

    st.caption(
        f"{result['role']} · {result['date']}"
    )

    a, b, c = st.columns(3)

    with a:
        show_metric("SEMANTIC SIMILARITY", f"{result['score']}%")

    with b:
        show_metric("MATCHED SKILLS", len(result["matched"]))

    with c:
        show_metric("SKILLS TO REVIEW", len(result["missing"]))

    st.write("")
    st.markdown("### 🎯 Text similarity indicator")
    st.progress(result["score"] / 100)

    st.caption(
        "Semantic similarity compares text meaning. "
        "It is not an official ATS score or a hiring probability."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 📈 Skill coverage")

        total = len(result["matched"]) + len(result["missing"])
        coverage = (
            len(result["matched"]) / total * 100
            if total else 0
        )

        chart_data = pd.DataFrame({
            "Category": ["Matched", "To review"],
            "Count": [
                len(result["matched"]),
                len(result["missing"])
            ]
        })

        fig = px.pie(
            chart_data,
            names="Category",
            values="Count",
            hole=0.68,
            color="Category",
            color_discrete_map={
                "Matched": "#34d399",
                "To review": "#c084fc"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=15, b=15, l=5, r=5),
            legend=dict(orientation="h", y=-0.1)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"pie_{result['id']}"
        )

        st.metric("Listed skill coverage", f"{coverage:.1f}%")

    with col2:
        st.markdown("### 🧩 Matched skills")

        if result["matched"]:
            html = "".join(
                f'<span class="pill">{skill.title()}</span>'
                for skill in result["matched"]
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No matching listed skills detected.")

        st.markdown("### 🔎 Skills to review")

        if result["missing"]:
            html = "".join(
                f'<span class="gap-pill">{skill.title()}</span>'
                for skill in result["missing"]
            )
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.success("No gaps found in the predefined skill list.")

    st.markdown("### 💡 Recommendations")

    if result["missing"]:
        st.warning(
            "Review the skills listed above. Include them in your "
            "resume only if you genuinely have that knowledge or experience."
        )
    else:
        st.success(
            "No gaps were detected in the predefined skill list. "
            "Make sure your projects and achievements are easy to find."
        )

    pdf_bytes = create_pdf_report(result)

    st.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"resumeiq_report_{result['id']}.pdf",
        mime="application/pdf",
        key=f"download_{result['id']}",
        use_container_width=True
    )

    if historical:
        st.caption("You are viewing a saved analysis from your history.")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 💎 ResumeIQ")
    st.caption("AI CAREER INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["✨ New Analysis", "🕘 Previous Analyses"],
        key="navigation"
    )

    st.divider()
    st.markdown("### Built with")
    st.caption("Python · NLP · SQLite")
    st.caption("Sentence Transformers · Plotly")
    st.caption("PyPDF · ReportLab")

    st.info(
        "Your history is saved locally in SQLite. "
        "The original resume PDF is not stored in the history database."
    )


# ---------------- NEW ANALYSIS PAGE ----------------
if page == "✨ New Analysis":

    st.markdown("""
    <div class="hero">
        <div style="letter-spacing:3px;color:#c4b5fd;font-weight:800">
            YOUR CAREER, UPGRADED
        </div>
        <h1>ResumeIQ AI ✨</h1>
        <p>
            Understand how your resume aligns with your target job.
            Discover matching skills and areas to improve.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.markdown("### 📄 Upload your resume")

        uploaded = st.file_uploader(
            "Choose a text-based PDF",
            type=["pdf"],
            key="resume_upload"
        )

        role = st.text_input(
            "Target job title",
            placeholder="Software Engineering Intern",
            key="target_role"
        )

    with right:
        st.markdown("### 🎯 Target job")

        job = st.text_area(
            "Job description",
            height=245,
            placeholder="Paste the complete job description here..."
        )

    if st.button(
        "✨ Run AI Analysis",
        type="primary",
        use_container_width=True
    ):
        if uploaded is None:
            st.error("Please upload your resume PDF.")

        elif not job.strip():
            st.error("Please paste the job description.")

        else:
            try:
                with st.spinner("Extracting and analyzing your resume..."):
                    resume_text = extract_text(uploaded)

                    if not resume_text.strip():
                        st.error(
                            "No readable text found. Try a text-based PDF."
                        )
                        st.stop()

                    score = semantic_score(resume_text, job)

                    resume_skills = find_skills(resume_text)
                    job_skills = find_skills(job)

                    matched = [
                        skill for skill in job_skills
                        if skill in resume_skills
                    ]

                    missing = [
                        skill for skill in job_skills
                        if skill not in resume_skills
                    ]

                    filename = uploaded.name
                    analyzed_at = datetime.now().strftime(
                        "%d %b %Y, %I:%M %p"
                    )

                    save_analysis(
                        role, filename, score, matched, missing
                    )

                    # Retrieve the saved record for its database ID
                    row = get_history()[0]

                    result = {
                        "id": row[0],
                        "date": analyzed_at,
                        "role": role.strip() or "Not specified",
                        "filename": filename,
                        "score": score,
                        "matched": matched,
                        "missing": missing,
                        "resume_skills": resume_skills
                    }

                    st.session_state["current_result"] = result

                st.toast("Analysis saved to Previous Analyses!")

            except Exception as error:
                st.error(
                    "Analysis failed. Check the terminal for details. "
                    "The first AI model download requires internet."
                )
                st.exception(error)

    if st.session_state.get("current_result"):
        show_result(st.session_state["current_result"])


# ---------------- HISTORY PAGE ----------------
elif page == "🕘 Previous Analyses":

    st.markdown("""
    <div class="hero">
        <div style="letter-spacing:3px;color:#c4b5fd;font-weight:800">
            YOUR SAVED RESULTS
        </div>
        <h1>Analysis History 🕘</h1>
        <p>
            Revisit your previous resume checks, compare results,
            and download reports again.
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = get_history()

    if not history:
        st.info(
            "No previous analyses yet. Go to New Analysis and "
            "analyze your first resume to start building your history."
        )

    else:
        st.markdown(f"### 📚 {len(history)} saved analyses")

        for row in history:
            (
                analysis_id,
                analyzed_at,
                saved_role,
                filename,
                saved_score,
                saved_matched,
                saved_missing
            ) = row

            matched_list = [
                s for s in (saved_matched or "").split(", ") if s
            ]

            missing_list = [
                s for s in (saved_missing or "").split(", ") if s
            ]

            with st.expander(
                f"📄 {saved_role}  ·  {saved_score}%  ·  {analyzed_at}"
            ):
                st.write(f"**Resume:** {filename or 'Not recorded'}")
                st.write(f"**Analysis ID:** {analysis_id}")

                a, b, c = st.columns(3)

                with a:
                    show_metric("SIMILARITY", f"{saved_score}%")

                with b:
                    show_metric("MATCHED SKILLS", len(matched_list))

                with c:
                    show_metric("SKILLS TO REVIEW", len(missing_list))

                st.markdown("**Matched skills**")
                st.write(", ".join(matched_list) or "None recorded")

                st.markdown("**Skills to review**")
                st.write(", ".join(missing_list) or "None recorded")

                historical_result = {
                    "id": analysis_id,
                    "date": analyzed_at,
                    "role": saved_role or "Not specified",
                    "filename": filename or "Not recorded",
                    "score": saved_score or 0,
                    "matched": matched_list,
                    "missing": missing_list
                }

                st.download_button(
                    "📥 Download this report",
                    data=create_pdf_report(historical_result),
                    file_name=f"resumeiq_report_{analysis_id}.pdf",
                    mime="application/pdf",
                    key=f"history_download_{analysis_id}"
                )

                if st.button(
                    "🗑️ Delete this record",
                    key=f"delete_{analysis_id}"
                ):
                    with sqlite3.connect(DB_NAME) as conn:
                        conn.execute(
                            "DELETE FROM analyses WHERE id = ?",
                            (analysis_id,)
                        )

                    st.rerun()

        st.caption(
            "History is stored locally on this computer. "
            "It includes analysis metadata and skill lists, "
            "not the original PDF or full job description."
        )

st.divider()
st.caption(
    "ResumeIQ AI · Semantic similarity is an estimate, "
    "not an official ATS score or hiring prediction."
)
