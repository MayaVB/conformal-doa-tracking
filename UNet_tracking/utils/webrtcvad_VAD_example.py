import numpy as np
import matplotlib.pyplot as plt
import wave
import webrtcvad

# Initialize VAD
VAD_Aggressiveness = 2
overlap_ms_param = 20
vad = webrtcvad.Vad(VAD_Aggressiveness)  # Aggressiveness level 0-3
title = 'waveform_and_vad_webrtcvad_sentence5'

# Function to read wave file
def read_wave(path):
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        assert num_channels == 1  # VAD works with mono audio
        assert sample_width == 2  # 16-bit audio
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def process_audio(audio_data, sample_rate, overlap_ms):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Ensure mono
    if audio_array.ndim > 1 and audio_array.shape[1] == 2:
        audio_array = audio_array.mean(axis=1).astype(np.int16)

    frame_length = int(sample_rate * 0.03)  # 30 ms frame
    overlap_length = int(sample_rate * (overlap_ms / 1000))  # Convert overlap_ms to sample count
    vad_results = []
    
    start = 0
    while start < len(audio_array):
        frame = audio_array[start:start + frame_length]
        if len(frame) < frame_length:
            break  # Skip incomplete frames
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        vad_results.extend([is_speech] * len(frame))
        
        start += frame_length - overlap_length  # Move start by frame length minus overlap

    # Match lengths of vad_results and audio_array
    if len(vad_results) < len(audio_array):
        vad_results.extend([0] * (len(audio_array) - len(vad_results)))
    else:
        vad_results = vad_results[:len(audio_array)]

    return audio_array, vad_results

# Read the audio data
audio_data, sr = read_wave('../dataset_folder/RevMovingSrcDatasetWavs/scene_2/01a/src_pos25/_ch2.wav')
with wave.open(f"{title}.wav", 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(audio_data)
    
audio_array, vad_results = process_audio(audio_data, sr, overlap_ms=overlap_ms_param)

# Create time axis for plotting
time = np.arange(len(audio_array)) / sr

# Plot waveform and VAD results
plt.figure(figsize=(12, 6))

# Plot waveform
plt.subplot(2, 1, 1)
plt.plot(time, audio_array)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot VAD results
plt.subplot(2, 1, 2)
plt.plot(time, vad_results, color='r')
plt.title("Voice Activity Detection")
plt.xlabel("Time (s)")
plt.ylabel("Speech Detected")
plt.ylim(-0.1, 1.1)  # Set limits for better visibility

plt.tight_layout()

# Save the figure as a JPEG file
plt.savefig(f"{title}_vad_aggressiveness_{VAD_Aggressiveness}.jpeg", format='jpeg', dpi=300)  # High quality
plt.close()  # Close the figure

print(f"Plot saved as '{title}_vad_aggressiveness_{VAD_Aggressiveness}.jpeg'")
