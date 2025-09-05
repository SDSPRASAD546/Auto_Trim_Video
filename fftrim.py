from moviepy.editor import VideoFileClip, concatenate_videoclips
import librosa
import numpy as np
import tempfile
import os

def trim_video_by_volume(
    input_path,
    output_path="output_trimmed.mp4",
    volume_threshold=0.5,     # compare to normalized RMS (0..1 of file's own max)
    margin=2.0,               # seconds before/after each loud region
    min_segment_length=0.7,   # drop super short blips after margins
    frame_length=2048,
    hop_length=512,
    smooth_win_seconds=0.25   # smooth RMS over ~250 ms
):
    # --- extract audio to a temp wav ---
    video = VideoFileClip(input_path)
    audio = video.audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
    audio.write_audiofile(audio_path, fps=44100)  # keep it simple; no verbose arg

    # --- load audio & compute RMS ---
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    os.remove(audio_path)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # --- smooth RMS (moving average over smooth_win_seconds) ---
    frames_per_second = sr / hop_length
    win_frames = max(1, int(smooth_win_seconds * frames_per_second))
    if win_frames > 1:
        kernel = np.ones(win_frames, dtype=float) / win_frames
        rms_smooth = np.convolve(rms, kernel, mode="same")
    else:
        rms_smooth = rms

    # --- normalize RMS to [0,1] so threshold like 0.5 means "50% of this file's loudest" ---
    max_rms = float(np.max(rms_smooth)) if np.any(rms_smooth > 0) else 1.0
    rms_norm = rms_smooth / (max_rms + 1e-9)

    # --- detect loud frames (normalized) ---
    loud = rms_norm >= float(volume_threshold)

    # --- collect contiguous loud regions in frame space, then convert to time ---
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
            seg_end_t = t + frame_dt  # include the current frame end
            loud_segments.append((seg_start_t, seg_end_t))

    if in_seg:
        loud_segments.append((seg_start_t, times[-1] + frame_dt))

    # --- expand by margins, clamp to duration ---
    expanded = [
        (max(0.0, s - margin), min(video.duration, e + margin))
        for (s, e) in loud_segments
    ]

    # --- merge overlaps after expansion to avoid duplicates/gaps ---
    expanded.sort(key=lambda x: x[0])
    merged = []
    for s, e in expanded:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # --- drop too-short segments ---
    final_segments = [(s, e) for (s, e) in merged if (e - s) >= min_segment_length]

    # --- cut and export ---
    if final_segments:
        clips = [video.subclip(s, e) for (s, e) in final_segments]
        out = concatenate_videoclips(clips, method="compose")
        out.write_videofile(output_path, codec="libx264", audio_codec="aac")
    else:
        print("No segments above threshold found.")

    # free resources
    video.close()
    if 'out' in locals():
        out.close()
# Example:
# Use 0.5 when you want the top half of the clip's loudness retained.
trim_video_by_volume(
    "11 kils.mp4",
    output_path="output_trimmed.mp4",
    volume_threshold=0.5,
    margin=2.0,
    min_segment_length=0.7
)
