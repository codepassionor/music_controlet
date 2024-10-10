import torch
from diffusers import DDPMPipeline


import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_audio_from_filename(filename):
    audio, sr = librosa.load(filename)
    D = np.abs(librosa.stft(audio))**2
    audio= librosa.feature.melspectrogram(y=audio, sr=sr, S=D)
    return audio
    
    
def convert_data(data_path):
    wav_filename = data_path
    audio = read_audio_from_filename(wav_filename)
    return audio


def generate_music(controlnet, genre_id, mood_id, controls, num_inference_steps, vocoder):
    controlnet.eval()
    
    with torch.no_grad():
        # Generate the mel spectrogram using the controlnet
        mel_spectrogram = controlnet(genre_id, mood_id, controls, num_inference_steps)
        
        # Convert the mel spectrogram to audio using the vocoder
        mel_spectrogram = convert_data(mel_spectrogram)
        audio = librosa.feature.inverse.mel_to_audio(mel_spectrogram)
    
    return audio

def evaluate_single_multiple_controls(controlnet, dataloader, device, num_inference_steps, vocoder):
    for batch in dataloader:
        genre_ids = batch["genre_ids"].to(device)
        mood_ids = batch["mood_ids"].to(device)
        controls = [control.to(device) for control in batch["controls"]]
        
        # Evaluate with single controls
        for control_idx in range(len(controls)):
            single_control = [controls[control_idx]]
            audio = generate_music(controlnet, genre_ids, mood_ids, single_control, num_inference_steps, vocoder)
            # Save or play the generated audio
            # ...
        
        # Evaluate with multiple controls
        audio = generate_music(controlnet, genre_ids, mood_ids, controls, num_inference_steps, vocoder)
        # Save or play the generated audio
        # ...
