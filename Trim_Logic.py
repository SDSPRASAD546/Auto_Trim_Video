from moviepy.editor import VideoFileClip, concatenate_videoclips
from proglog import ProgressBarLogger
import librosa
import numpy as np
import tempfile
import os

class CallbackLogger(ProgressBarLogger):
    def __init__(self, cb):
        super().__init__()
        self.cb = cb
    def bars_callback(self, bar, attr, value, old_value=None):
        if bar == "t":
            total = self.bars[bar].get("total") or 0
            if total:
                p = int((value / total) * 100)
                self.cb(max(0, min(100, p)))

def trim_video_by_volume(
    input_path,
    output_path="output_trimmed.mp4",
    volume_threshold=0.5,
    margin=2.0,
    min_segment_length=0.7,
    frame_length=2048,
    hop_length=512,
    smooth_win_seconds=0.25,
    progress_cb=None
):
    video = VideoFileClip(input_path)
    audio = video.audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
    audio.write_audiofile(audio_path, fps=44100)
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

    loud_segments = []
    in_seg = False
    seg_start_t = None
    frame_dt = hop_length / sr

    for i, is_loud in enumerate(loud):
        t = times[i]
        if is_loud and not in_seg:
            in_seg = True
            seg_start_t = t
        elif not is_loud and in_seg:
            in_seg = False
            seg_end_t = t + frame_dt
            loud_segments.append((seg_start_t, seg_end_t))
    if in_seg:
        loud_segments.append((seg_start_t, times[-1] + frame_dt))

    expanded = [(max(0.0, s - margin), min(video.duration, e + margin)) for (s, e) in loud_segments]
    expanded.sort(key=lambda x: x[0])
    merged = []
    for s, e in expanded:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    final_segments = [(s, e) for (s, e) in merged if (e - s) >= min_segment_length]

    if progress_cb:
        progress_cb(0)
    if final_segments:
        clips = [video.subclip(s, e) for (s, e) in final_segments]
        out = concatenate_videoclips(clips, method="compose")
        logger = CallbackLogger(progress_cb) if progress_cb else None
        out.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=logger)
        out.close()
    video.close()
    if progress_cb:
        progress_cb(100)
