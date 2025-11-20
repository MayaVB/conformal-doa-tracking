import numpy as np
import matplotlib.pyplot as plt
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import wave
import torch  # Import if using PyTorch

title = 'waveform_and_vad_silero_sentence5'
def read_wave(path):
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        assert num_channels == 1  # VAD works with mono audio
        assert sample_width == 2  # 16-bit audio
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

# Load the VAD model
model = load_silero_vad()

# Read the audio data
wav = read_audio('../dataset_folder/RevMovingSrcDatasetWavs/scene_2/01a/src_pos25/_ch2.wav') 

audio_data, sr = read_wave('../dataset_folder/RevMovingSrcDatasetWavs/scene_2/01a/src_pos25/_ch2.wav')  
with wave.open(f"{title}.wav", 'wb') as wf: wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(audio_data)

# Get speech timestamps
speech_timestamps = get_speech_timestamps(wav, model)

# Debugging: print the speech timestamps
print("Speech Timestamps:", speech_timestamps)

# Create a time axis for plotting
time = np.arange(len(wav)) / 16000  # Assuming a sample rate of 16 kHz

# Create a binary array for speech detection
vad_binary = np.zeros(len(wav))

# Process speech timestamps
for timestamp in speech_timestamps:
    start = int(timestamp['start'])
    end = int(timestamp['end'])
    vad_binary[start:end] = 1  # Convert to integers and set speech detected

# Plot waveform and VAD results
plt.figure(figsize=(12, 6))

# Plot waveform
plt.subplot(2, 1, 1)
plt.plot(time, wav)
plt.title("Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot VAD results
plt.subplot(2, 1, 2)
plt.plot(time, vad_binary, color='r')
plt.title("Voice Activity Detection (Silero VAD)")
plt.xlabel("Time (s)")
plt.ylabel("Speech Detected")
plt.ylim(-0.1, 1.1)  # Set limits for better visibility

plt.tight_layout()

# Save the figure as a JPEG file
plt.savefig(f"{title}.jpeg", format='jpeg', dpi=300)  # High quality
plt.close()  # Close the figure
# print("Plot saved as 'waveform_and_vad_silero.jpeg'")



# from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# model = load_silero_vad()
# wav = read_audio('../dataset_folder/RevMovingSrcDatasetWavs/scene_2/01a/src_pos15/_ch1.wav') # backend (sox, soundfile, or ffmpeg) required!
# speech_timestamps = get_speech_timestamps(wav, model)