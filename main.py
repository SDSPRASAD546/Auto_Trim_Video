from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, concatenate_videoclips
import librosa
import numpy as np
import tempfile
import os
import threading
import time

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.secret_key = "supersecret"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

progress_data = {"progress": 0, "status": "idle"}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Trim Function ---
def trim_video_by_volume(
    input_path,
    output_path="output_trimmed.mp4",
    volume_threshold=0.5,
    margin=2.0,
    min_segment_length=0.7,
    frame_length=2048,
    hop_length=512,
    smooth_win_seconds=0.25
):
    video = VideoFileClip(input_path)
    audio = video.audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
    audio.write_audiofile(audio_path, fps=44100, logger=None)


    y, sr = librosa.load(audio_path, sr=None, mono=True)
    os.remove(audio_path)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    frames_per_second = sr / hop_length
    win_frames = max(1, int(smooth_win_seconds * frames_per_second))
    if win_frames > 1:
        kernel = np.ones(win_frames, dtype=float) / win_frames
        rms_smooth = np.convolve(rms, kernel, mode="same")
    else:
        rms_smooth = rms

    max_rms = float(np.max(rms_smooth)) if np.any(rms_smooth > 0) else 1.0
    rms_norm = rms_smooth / (max_rms + 1e-9)
    loud = rms_norm >= float(volume_threshold)

    loud_segments, in_seg, seg_start_t = [], False, None
    frame_dt = hop_length / sr
    for i, is_loud in enumerate(loud):
        t = times[i]
        if is_loud and not in_seg:
            in_seg, seg_start_t = True, t
        elif not is_loud and in_seg:
            in_seg = False
            loud_segments.append((seg_start_t, t + frame_dt))
    if in_seg:
        loud_segments.append((seg_start_t, times[-1] + frame_dt))

    expanded = [
        (max(0.0, s - margin), min(video.duration, e + margin))
        for (s, e) in loud_segments
    ]

    expanded.sort(key=lambda x: x[0])
    merged = []
    for s, e in expanded:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    final_segments = [(s, e) for (s, e) in merged if (e - s) >= min_segment_length]

    if final_segments:
        clips = [video.subclip(s, e) for (s, e) in final_segments]

        out = concatenate_videoclips(clips, method="compose")

        # Fake progress simulation (MoviePy doesn’t expose fine-grained progress easily)
        for i in range(1, 101):
            time.sleep(0.1)  # simulate work
            progress_data["progress"] = i

        out.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None, verbose=False)
        out.close()
    else:
        print("No segments above threshold found.")
    video.close()


def run_trimming(input_path, output_path):
    progress_data["progress"] = 0
    progress_data["status"] = "processing"
    trim_video_by_volume(input_path, output_path=output_path)
    progress_data["progress"] = 100
    progress_data["status"] = "done"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash("No video uploaded.")
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash("No selected file.")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            output_name = "trimmed_" + filename
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)

            threading.Thread(target=run_trimming, args=(input_path, output_path)).start()

            return render_template("result.html", filename=output_name)

        else:
            flash("Invalid file format. Only MP4 allowed.")
            return redirect(request.url)

    return render_template('upload.html')


# 🔥 Server-Sent Events for real-time progress
@app.route('/progress_stream')
def progress_stream():
    def generate():
        last_progress = -1
        while True:
            if progress_data["progress"] != last_progress:
                last_progress = progress_data["progress"]
                yield f"data: {progress_data['progress']}\n\n"
            if progress_data["status"] == "done":
                break
            time.sleep(0.5)
    return Response(generate(), mimetype='text/event-stream')


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
