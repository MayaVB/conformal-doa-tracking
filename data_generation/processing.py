import numpy as np
import matplotlib.pyplot as plt

def generate_ar1_noise(length, decay):
    """Generate AR(1) noise."""
    noise = np.zeros(length)
    for i in range(1, length):
        noise[i] = decay * noise[i - 1] + np.random.normal(0, 1)  # Add Gaussian noise
    return noise

def add_ar1_noise_to_signals(signals, decay, snr_db):
    """Add AR(1) noise to each microphone signal based on SNR."""
    noisy_signals = np.zeros_like(signals)  # Initialize an array for noisy signals
    for mic in range(signals.shape[0]):  # Loop through each microphone
        noise = generate_ar1_noise(signals.shape[1], decay)  # Generate AR(1) noise
        
        # Calculate signal power and noise power based on desired SNR
        signal_power = np.mean(signals[mic, :] ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))  # Convert SNR from dB to linear scale
        
        # Scale noise to achieve desired SNR
        noise_scale = np.sqrt(noise_power / np.mean(noise ** 2))
        noisy_signals[mic, :] = signals[mic, :] + noise * noise_scale  # Add scaled noise to the signal
    return noisy_signals