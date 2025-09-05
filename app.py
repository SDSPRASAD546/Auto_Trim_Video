import os
import uuid
import time
import threading
from threading import Thread
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from Trim_Logic import trim_video_by_volume

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

JOBS = {}  # global job store

CLEANUP_INTERVAL = 60        # seconds
TIMEOUT_SECONDS = 15 * 60    # 15 minutes


def _run_trim(job_id, input_path, output_path, margin):
    def cb(progress_percent):
        JOBS[job_id]["progress"] = int(progress_percent)
        JOBS[job_id]["last_update"] = time.time()

    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["start_time"] = time.time()
    JOBS[job_id]["last_update"] = time.time()
    try:
        trim_video_by_volume(input_path, output_path, margin=margin, progress_cb=cb)
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["download"] = os.path.basename(output_path)
        JOBS[job_id]["finish_time"] = time.time()
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["finish_time"] = time.time()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "error": "No selected file"}), 400

    margin = float(request.form.get("margin", 2.0))
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    output_name = f"trimmed_{filename}"
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_name)
    file.save(input_path)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "input": input_path,
        "output": output_path,
        "start_time": time.time(),
        "last_update": time.time()
    }

    t = Thread(target=_run_trim, args=(job_id, input_path, output_path, margin), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job id"}), 404

    resp = {"status": job["status"], "progress": job.get("progress", 0)}
    if job["status"] == "done":
        if os.path.exists(job["output"]):
            resp["download_url"] = url_for("download_file", filename=job["download"])
            resp["video_url"] = url_for("download_file", filename=job["download"])
        else:
            resp["status"] = "expired"
            resp["error"] = "Video has timed out and was deleted."
    if job["status"] == "error":
        resp["error"] = job.get("error", "Unknown error")
    return jsonify(resp)


@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.exists(path):
        return "File expired", 404
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=False)


def cleanup_worker():
    while True:
        now = time.time()
        for job_id, job in list(JOBS.items()):
            if job["status"] in ("done", "error"):
                finish = job.get("finish_time", job.get("last_update", now))
                if now - finish > TIMEOUT_SECONDS:
                    # delete files
                    for f in (job.get("input"), job.get("output")):
                        if f and os.path.exists(f):
                            try:
                                os.remove(f)
                            except:
                                pass
                    # mark job expired
                    JOBS[job_id]["status"] = "expired"
        time.sleep(CLEANUP_INTERVAL)


# start cleanup thread
threading.Thread(target=cleanup_worker, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True)
