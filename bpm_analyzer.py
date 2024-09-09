import streamlit as st
import os
import subprocess
from moviepy.editor import VideoFileClip
import librosa
import pandas as pd
import numpy as np
import shutil

# Function to get video duration using ffprobe
def get_video_duration(video_file):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

def calculate_bpm_for_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

# Function to trim video using ffmpeg
def trim_video_ffmpeg(video_file, trimmed_video_file, trim_duration=3.5 * 60):
    video_duration = get_video_duration(video_file)
    
    end_time = video_duration - trim_duration
    
    command = [
        'ffmpeg',
        '-i', video_file,
        '-ss', '0',
        '-to', str(end_time),
        '-c', 'copy',
        trimmed_video_file
    ]
    subprocess.run(command)

# Function to extract audio from video using MoviePy
def extract_audio_from_video(video_file, audio_output):
    video = VideoFileClip(video_file)
    video.audio.write_audiofile(audio_output)

# Function to detect music segments by silence using librosa
def detect_music_by_silence(audio_file, silence_threshold=0.02, min_duration=25.0, min_silence_duration=2.0):
    y, sr = librosa.load(audio_file, sr=None)
    
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=512)
    
    non_silent = energy > silence_threshold
    segments = []
    start_time = None
    
    for t, is_non_silent in zip(times, non_silent):
        if is_non_silent and start_time is None:
            start_time = t
        elif not is_non_silent and start_time is not None:
            segment_duration = t - start_time
            if segment_duration >= min_duration:
                segments.append((start_time, t))
            start_time = None
    
    if start_time is not None:
        segment_duration = times[-1] - start_time
        if segment_duration >= min_duration:
            segments.append((start_time, times[-1]))
    
    filtered_segments = []
    for i in range(len(segments) - 1):
        if segments[i+1][0] - segments[i][1] >= min_silence_duration:
            filtered_segments.append(segments[i])
    
    return filtered_segments

# Function to calculate BPM for a music segment
def calculate_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

# Function to calculate BPM for all music segments
def get_bpm_for_music_sections(audio_file, music_segments):
    y, sr = librosa.load(audio_file, sr=None)
    bpm_list = []

    for segment in music_segments:
        start_time = segment[0]
        end_time = segment[-1] if len(segment) > 1 else segment[0] + 1 
        start_sample = librosa.time_to_samples(start_time, sr=sr)
        end_sample = librosa.time_to_samples(end_time, sr=sr)

        y_segment = y[start_sample:end_sample]
        bpm = calculate_bpm(y_segment, sr)
        bpm_list.append(bpm)

    return bpm_list

# Function to calculate weighted average BPM
def calculate_weighted_average_bpm(bpm_values, segments):
    total_duration = 0
    weighted_bpm_sum = 0
    
    for bpm, segment in zip(bpm_values, segments):
        start_time = segment[0]
        end_time = segment[-1] if len(segment) > 1 else segment[0] + 1
        duration = end_time - start_time
        
        weighted_bpm_sum += bpm * duration
        total_duration += duration
    
    weighted_average_bpm = weighted_bpm_sum / total_duration if total_duration > 0 else 0
    return weighted_average_bpm

# Streamlit App
st.title("Video BPM Analyzer")

# Initialize session state to track the analysis type
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None

# Let the user select the type of analysis
analysis_type = st.radio(
    "Choose the type of analysis:",
    ("Full Episode Analysis", "Song Video Analysis")
)

# If the analysis type changes, clear the uploaded files
if analysis_type != st.session_state.analysis_type:
    st.session_state.uploaded_files = []
    st.session_state.analysis_type = analysis_type

# Upload multiple files (to simulate folder upload)
uploaded_files = st.file_uploader(
    "Upload video files (select multiple files to simulate folder upload)", 
    type=["mp4", "mov", "avi"], 
    accept_multiple_files=True
)

# Store the uploaded files in a list in session state
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Button to run the analysis
if st.button("Run Analysis") and st.session_state.uploaded_files:
    # Create a temporary directory to save uploaded files
    with st.spinner("Processing..."):
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded files
        video_files = []
        for file in st.session_state.uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
            video_files.append(file_path)

        results = []

        if analysis_type == "Full Episode Analysis":
            for video_file in video_files:
                # Process the video using the first method
                trimmed_video_file = os.path.join(temp_dir, f"trimmed_{os.path.basename(video_file)}")
                audio_output = os.path.join(temp_dir, f"extracted_audio_{os.path.splitext(os.path.basename(video_file))[0]}.wav")
                
                # Simulate the trimming function (you can implement it if needed)
                extract_audio_from_video(video_file, audio_output)
                
                # Start with a max duration and decrement until a valid BPM is found or the limit is reached
                min_duration = 50
                found_bpm = False
                weighted_average_bpm = 0

                while min_duration > 0 and not found_bpm:
                    music_segments = detect_music_by_silence(audio_output, min_duration=min_duration)
                    bpm_values = get_bpm_for_music_sections(audio_output, music_segments)
                    weighted_average_bpm = calculate_weighted_average_bpm(bpm_values, music_segments)
                    if weighted_average_bpm > 0:
                        found_bpm = True
                    else:
                        min_duration -= 10
                
                if found_bpm:
                    results.append({
                        "Video Title": os.path.basename(video_file),
                        "Weighted Average BPM": weighted_average_bpm
                    })

        elif analysis_type == "Song Video Analysis":
            for video_file in video_files:
                audio_output = os.path.join(temp_dir, f"extracted_audio_{os.path.splitext(os.path.basename(video_file))[0]}.wav")
                extract_audio_from_video(video_file, audio_output)
                bpm = calculate_bpm_for_audio(audio_output)
                results.append({
                    "Video Title": os.path.basename(video_file),
                    "BPM": bpm
                })

        # Convert results to DataFrame and display
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Cleanup
        for file in video_files:
            os.remove(file)

        # Remove the entire directory and its contents
        shutil.rmtree(temp_dir)

    st.success("Done!")
