import os
import fitz
from flask import Flask, flash, redirect, render_template, request, url_for
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")
app.config["UPLOAD_FOLDER"] = os.path.join(app.instance_path, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


def extract_text_from_pdf(pdf_path):
    text_blocks = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_blocks.append(page.get_text())
    return "\n".join(text_blocks).strip()


def compute_similarity(resume_text, job_desc):
    embeddings = model.encode(
        [resume_text, job_desc],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity) * 100, 2)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", score=None, job_desc="")


@app.route("/score", methods=["POST"])
def score():
    if "resume" not in request.files:
        flash("Please upload a resume PDF file.")
        return redirect(url_for("index"))

    resume_file = request.files["resume"]
    job_desc = request.form.get("job_desc", "").strip()

    if resume_file.filename == "" or not job_desc:
        flash("Resume and job description are both required.")
        return redirect(url_for("index"))

    if not allowed_file(resume_file.filename):
        flash("Only PDF resumes are supported.")
        return redirect(url_for("index"))

    filename = secure_filename(resume_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    resume_file.save(filepath)

    try:
        resume_text = extract_text_from_pdf(filepath)
        if not resume_text:
            flash("Unable to extract text from the uploaded resume.")
            return redirect(url_for("index"))

        score_value = compute_similarity(resume_text, job_desc)
        return render_template(
            "index.html",
            score=score_value,
            resume_filename=filename,
            job_desc=job_desc,
        )
    except Exception as exc:
        flash(f"Error processing resume: {exc}")
        return redirect(url_for("index"))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True)
