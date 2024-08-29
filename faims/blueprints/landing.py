from flask import Blueprint, current_app, render_template, session, redirect, url_for, request, flash, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
from faims.db import get_db, get_unique_value_for_field
import json
from os.path import join
import os
from faims.analysis import get_peptide_max_length, faims_cv_prediction, get_prediction_label
import numpy as np
import zipfile
import tempfile


bp = Blueprint("landing", __name__, url_prefix="/")


@bp.route("/", methods=["GET", "POST"])
def index():
    # Check if user has submitted jobs
    if request.method == "GET":
        if "id_peptides" in session.keys():
            # Get jobs
            id_peptides = session["id_peptides"]
            db = get_db()
            # Four possibilities for status: queued, error, running, completed
            jobs = {}
            jobs["queued"] = []
            jobs["error"] = []
            jobs["running"] = []
            jobs["completed"] = []
            for id_peptides in id_peptides:
                for stat in ("queued", "error", "running"):
                    j = db.execute("SELECT id_peptides, job_name, creation_time FROM queue WHERE status = ? AND id_peptides = ?", (stat, id_peptides)).fetchone()
                    if j is not None:
                        jobs[stat].append(j)

                j = db.execute("SELECT q.id_peptides, q.job_name, r.id_result, r.creation_time "
                                                 "FROM queue AS q INNER JOIN results AS r ON q.id_peptides = r.id_peptides "
                                                 "WHERE q.status = ? AND q.id_peptides = ?", ("completed", id_peptides)).fetchone()
                if j is None:
                    continue
                j = dict(j)
                jobs["completed"].append(dict(j))
            jobs["completed"].sort(key=lambda x: x["creation_time"], reverse=True)
            return render_template("index.html", jobs=jobs, id_peptides=session["id_peptides"])
        return render_template("index.html", jobs=None)
    elif request.method == "POST":
        peptides = parse_peptides(request.form["peptides"])
        are_valid = validate_peptides(peptides)
        if not are_valid:
            return redirect(url_for("landing.index"))
        job_name = request.form["job_name"][:200]
        # Save peptides to file
        id_peptides = save_peptides(peptides)
        add_peptide_to_session(id_peptides)

        # Submit job
        submit_faims_job(id_peptides, job_name=job_name)

        # Run analysis if possible, save results
        id_result = run_analysis(id_peptides=id_peptides, peptides=peptides)
        # if not is_analysis_running():
        #     id_result = run_analysis(id_peptides=id_peptides, peptides=peptides)
        # else:
        #     flash("Job has been queued for analysis.")
        #     msg = "queued"
        #     return jsonify({"msg": msg})
        return redirect(url_for("landing.index"))


@bp.route("/results/<id_peptides>", methods=["GET"])
def download_file(id_peptides):
    if id_peptides in session["id_peptides"]:
        db = get_db()
        descriptor = db.execute("SELECT r.id_result, q.job_name FROM results AS r INNER JOIN queue AS q ON r.id_peptides = q.id_peptides WHERE r.id_peptides = ?", (id_peptides,)).fetchone()
        id_result = descriptor["id_result"]
        job_name = descriptor["job_name"]


        out_prefix = join(current_app.config["RESULTS_DIRECTORY"], id_result)
        pred_file = out_prefix + "_predictions.npy"
        pred_labels_file = out_prefix + "_prediction_labels"

        tmp_name = tempfile.mkstemp(suffix=".zip")[-1]
        z = zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_STORED)
        z.write(pred_file, arcname="predictions.npy")
        z.write(pred_labels_file, arcname="prediction_labels.txt")
        z.close()

        @after_this_request
        def remove_file(response):
            os.remove(tmp_name)
            return response

        return send_file(tmp_name, download_name=f"{secure_filename(job_name)}.zip", as_attachment=True)


def get_next_peptide_id():
    db = get_db()
    next_peptides = db.execute("SELECT id_peptides FROM queue WHERE status = ? ORDER BY creation_time ASC LIMIT 1 ", ("queued",)).fetchone()
    if next_peptides is None:
        return None
    return next_peptides["id_peptides"]


def is_analysis_running():
    db = get_db()
    if db.execute("SELECT id_peptides FROM queue WHERE status = ?", ("running",)).fetchone() is not None:
        return True
    return False


def save_results(id_peptides, predictions, prediction_labels) -> str:
    """Writes the results to the DB and returns the result ID"""
    db = get_db()
    id_result = get_unique_value_for_field(db, field="id_result", table="results")
    out_prefix = join(current_app.config["RESULTS_DIRECTORY"], id_result)
    while os.path.exists(out_prefix + "_predictions"):
        id_result = get_unique_value_for_field(db, field="id_result", table="results")
        out_prefix = join(current_app.config["RESULTS_DIRECTORY"], id_result)
    pred_file = out_prefix + "_predictions.npy"
    pred_labels_file = out_prefix + "_prediction_labels"
    np.save(pred_file, predictions)
    f = open(pred_labels_file, "w")
    for label in prediction_labels:
        f.write(f"{label}\n")
    f.close()
    ## Zip the two files together (makes the total size larger, so meh)
    # z = zipfile.ZipFile(out_prefix + ".zip", "w")
    # z.write(pred_file)
    # z.write(pred_labels_file)
    # z.close()

    # Log into db
    db.execute("INSERT INTO results (id_result, id_peptides) VALUES (?,?)", (id_result, id_peptides))
    db.execute("UPDATE queue SET status = ? WHERE id_peptides = ?", ("completed", id_peptides))
    db.commit()

    return id_result


def run_analysis(id_peptides: str = None, peptides: list = None):
    next_id_peptides = get_next_peptide_id()
    if id_peptides != next_id_peptides or peptides is None:
        # Load new peptide list
        peptides = load_peptides(next_id_peptides)
    db = get_db()
    db.execute("UPDATE queue SET status = ? WHERE id_peptides = ?", ("running", next_id_peptides))
    db.commit()
    try:
        predictions, prediction_labels = analyze_peptides(peptides)
        id_result = save_results(next_id_peptides, predictions, prediction_labels)
    except Exception as e:
        db.execute("UPDATE queue SET status = ? WHERE id_peptides = ?", ("error", next_id_peptides))
        db.commit()
        flash("An error occurred during processing")
        raise e

    return id_result
    # Check if there are other elements in the queue
    # if get_next_peptide_id() is None or is_analysis_running():
    #     return id_result
    # else:
    #     pid = os.fork()
    #     if pid > 0:
    #         # Parent process; just return
    #         return id_result
    #     else:
    #         # Child process: run next analysis
    #         return run_analysis()


def analyze_peptides(peptides: list = None):
    """Analyzes the specified peptides."""
    predictions = faims_cv_prediction(peptides, current_app.config["MODEL_FILE"])
    prediction_labels = get_prediction_label(predictions)
    return predictions, prediction_labels


def submit_faims_job(id_peptides, job_name=None):
    db = get_db()
    db.execute("INSERT INTO queue (id_peptides, job_name, status) VALUES(?, ?, ?)", (id_peptides, job_name, "queued"))
    db.commit()
    return


def load_peptides(id_peptides) -> list:
    """Loads the previously-saved peptide list."""
    f = open(join(current_app.config["UPLOAD_DIRECTORY"], id_peptides), "r")
    peptides = f.read().splitlines()
    f.close()
    return peptides


def save_peptides(peptides):
    db = get_db()
    id_peptides = get_unique_value_for_field(db, field="id_peptides", table="queue")
    out_file = join(current_app.config["UPLOAD_DIRECTORY"], id_peptides)
    while os.path.exists(out_file):
        id_peptides = get_unique_value_for_field(db, field="id_peptides", table="queue")
        out_file = join(current_app.config["UPLOAD_DIRECTORY"], id_peptides)
    f = open(join(current_app.config["UPLOAD_DIRECTORY"], id_peptides), "w")
    for pep in peptides:
        f.write(f"{pep}\n")
    f.close()
    return id_peptides


def parse_peptides(peptides: str,
                   delimiters: list = ("\n", ",", " ")) -> list:
    """Parses str of peptides and returns it as a list."""
    peptide_list = peptides.split(delimiters[0])
    for delim in delimiters[1:]:
        new_peptide_list = []
        for pep in peptide_list:
            pep = pep.strip()
            if len(pep) == 0:
                continue
            new_peptide_list += pep.split(delim)
        peptide_list = new_peptide_list
    return peptide_list


def validate_peptides(peptides: list) -> bool:
    if len(peptides) > current_app.config["MAX_PEPTIDE_COUNT"]:
        flash(f"Too many submitted peptides; reduce to a maximum of {current_app.config['MAX_PEPTIDE_COUNT']}")
        return False
    submitted_max_len = get_peptide_max_length(peptides)
    if submitted_max_len > current_app.config["MAX_PEPTIDE_LENGTH"]:
        flash(f"Maximum peptide length exceeded; reduce to a maximum of {current_app.config['MAX_PEPTIDE_LENGTH']}")
        return False
    return True


def add_peptide_to_session(id_peptides: str):
    session.permanent = True
    if "id_peptides" not in session.keys():
        session["id_peptides"] = []
    session["id_peptides"].append(id_peptides)
    session.modified = True
    return

@bp.route("/about", methods=["GET"])
def about():
    return render_template("about.html")
