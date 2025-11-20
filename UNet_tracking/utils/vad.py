import numpy as np


def compute_time_freq_spp(stft, threshold=0.5, low_freq_emphasis=True):
    """
    Compute a time-frequency VAD mask based on energy, with optional low-frequency emphasis.
    
    Parameters:
    - stft: The STFT of the speech signal (magnitude spectrogram).
    - threshold: Threshold for detecting speech (0 < threshold < 1).
    - low_freq_emphasis: Whether to emphasize low frequencies for VAD.
    
    Returns:
    - vad_mask: Binary mask indicating presence of speech (1: speech, 0: non-speech) for each time-frequency bin.
    """
    vad_mask = []
    # Compute the energy of the STFT (magnitude squared)
    for mic in np.arange(stft.shape[2]):
        stft_mic = stft[:, :, mic]
        energy = np.abs(stft_mic) ** 2
        
        # Optionally emphasize low frequencies by applying a frequency weighting
        if low_freq_emphasis:
            # Apply a simple weighting to the low-frequency bins (e.g., emphasizing the first 100 Hz)
            freq_bins = np.linspace(0, stft_mic.shape[0], stft_mic.shape[0])
            low_freq_weight = np.exp(-freq_bins / 50)  # Exponential decay for low frequencies
            energy = energy * low_freq_weight[:, np.newaxis]
        
        # Compute the mean energy across all frames for each frequency bin
        mean_energy = np.mean(energy, axis=1)  # Mean energy for each frequency across all frames
        
        # Apply threshold to classify speech vs. non-speech for each frequency bin and frame
        vad_mask_mic = energy > (threshold * mean_energy[:, np.newaxis])  # Compare energy per bin/frame
        vad_mask.append(vad_mask_mic)
        
    vad_mask = np.stack(vad_mask, axis=2)  

    return vad_mask


def compute_spp_mask(stft, noise_estimation_method='spectral_subtraction', threshold=0.5, alpha=0.9):
    """
    Compute the Signal Probability Presence (SPP) mask using spectral subtraction for noise estimation.
    
    Parameters:
    - stft: The STFT of the speech signal (magnitude spectrogram), already cut to the desired frequency range (0-400 Hz).
    - noise_estimation_method: Method for noise estimation ('spectral_subtraction' or 'wiener_filter').
    - threshold: Threshold for detecting speech (0 < threshold < 1).
    - alpha: Smoothing factor for noise estimation.
    
    Returns:
    - spp_mask: Binary mask indicating presence of speech (1: speech, 0: non-speech).
    """
     # Step 1: Estimate noise spectrum using spectral subtraction or Wiener filter
    if noise_estimation_method == 'spectral_subtraction':
        # Median estimate of noise across time (basic spectral subtraction)
        noise_estimate = np.median(np.abs(stft), axis=1, keepdims=True)
        speech_signal = np.abs(stft) - noise_estimate
        speech_signal[speech_signal < 0] = 0  # Avoid negative values
    
    elif noise_estimation_method == 'wiener_filter':
        # Wiener filtering approach using exponential smoothing for noise estimation
        noise_estimate = np.zeros_like(stft)
        # Exponentially smoothing the noise estimate across time
        for t in range(1, stft.shape[1]):
            noise_estimate[:, t] = alpha * noise_estimate[:, t-1] + (1 - alpha) * np.abs(stft[:, t])
        speech_signal = np.maximum(np.abs(stft) - noise_estimate, 0)
    
    # Step 2: Compute logarithmic SNR
    snr = np.log10(np.abs(speech_signal) / (noise_estimate + 1e-2))  # Add small value to avoid division by zero
    
    # Step 3: Threshold to create the SPP mask (speech = 1, non-speech = 0)
    spp_mask = snr > threshold
    
    return spp_mask
