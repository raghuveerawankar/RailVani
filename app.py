import streamlit as st
import pandas as pd
import asyncio
from zyphra import AsyncZyphraClient
import whisper
import jiwer
import librosa
import numpy as np
from pystoi import stoi
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

API_KEY = "zsk-1828271b9905fc6fd01835903489add6af1e35dda073a4eafd5fd97adc187276"  # Replace with your Zyphra API key 

async def generate_speech(text, output_path):
    async with AsyncZyphraClient(api_key=API_KEY) as client:
        await client.audio.speech.create(
            text=text,
            speaking_rate=15,
            output_path=output_path,
            audio_format="wav"
        )
        return output_path

def transcribe_audio(audio_path):
    model = whisper.load_model("large")  # Use "large" for better accuracy
    result = model.transcribe(audio_path)
    return result["text"]

def compute_mcd(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)  # Ensure 16kHz sample rate
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    distance, _ = fastdtw(mfcc.T, mfcc.T, dist=euclidean)
    return distance / len(mfcc.T)

def extract_pitch_librosa(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    pitches, voiced_flags, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitches = pitches[voiced_flags]
    return np.std(pitches)

def evaluate_audio(reference_text, audio_path):
    hypothesis_text = transcribe_audio(audio_path)
    wer = jiwer.wer(reference_text, hypothesis_text)
    cer = jiwer.cer(reference_text, hypothesis_text)
    gen_audio, sr = librosa.load(audio_path, sr=16000)
    clean_audio = librosa.effects.preemphasis(gen_audio)
    stoi_score = stoi(clean_audio, gen_audio, sr, extended=False)
    mcd_score = compute_mcd(audio_path)
    pitch_variability = extract_pitch_librosa(audio_path)
    return [wer, cer, stoi_score, mcd_score, pitch_variability]

def main():
    st.title("Train Announcement System with Evaluation")
    
    train_number = st.text_input("Train Number (6 Digits)")
    source_station = st.text_input("Source Station Name")
    destination_station = st.text_input("Destination Station Name")
    train_name = st.text_input("Train Name")
    platform_number = st.number_input("Platform Number", min_value=1, step=1, format="%d")
    arrival_time = st.time_input("Arrival Time")
    am_pm = st.selectbox("AM/PM", ["AM", "PM"])
    
    if st.button("Generate Announcement"):
        if not train_number.isdigit() or len(train_number) != 6:
            st.error("Please enter a valid 6-digit train number.")
            return
        
        formatted_train_number = ", ".join(train_number)
        formatted_arrival_time = arrival_time.strftime("%I:%M") + f" {am_pm}"
        
        announcement_text = (f"Your Attention Please! Train number {formatted_train_number}, from {source_station}, "
                             f"to {destination_station}, {train_name}, is arriving on Platform Number "
                             f"{platform_number} on its scheduled arrival time {formatted_arrival_time}.")
        
        st.success(announcement_text)
        
        audio_paths = []
        
        eval_results = []
        for i in range(1, 2):
            output_path = f"output_async_{i}.wav"
            asyncio.run(generate_speech(announcement_text, output_path))
            audio_paths.append(output_path)
        
        for i, path in enumerate(audio_paths, 1):
            st.audio(path, format="audio/wav", start_time=0)
            eval_results.append(evaluate_audio(announcement_text, path))
        
        df_eval = pd.DataFrame(eval_results, columns=["WER", "CER", "STOI Score", "MCD Score", "Pitch Variability"],
                               index=["Announcement 1", "Announcement 2", "Announcement 3"])
        
        st.write("### Evaluation Results")
        st.table(df_eval)

if __name__ == "__main__":
    main()
