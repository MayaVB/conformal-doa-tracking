import numpy as np
from itertools import combinations
from numpy.linalg import norm
import matplotlib.pyplot as plt

def parabolic_interpolation(lags, values):
    """
    Perform parabolic interpolation around the peak of a discrete signal
    to estimate a more accurate location of the peak.

    Parameters:
        lags (np.ndarray): Array of lag values corresponding to the signal.
        values (np.ndarray): Array of signal values for interpolation.

    Returns:
        float: Interpolated peak location.
    """
    # Find the index of the maximum value
    peak_index = np.argmax(values)
    
    # Extract the three points around the peak
    peak_values = np.array([values[peak_index - 1], values[peak_index], values[peak_index + 1]])
    peak_lags = np.array([[lags[peak_index - 1]**2, lags[peak_index - 1], 1],
                          [lags[peak_index]**2, lags[peak_index], 1],
                          [lags[peak_index + 1]**2, lags[peak_index + 1], 1]])
    
    # Solve for the parabola coefficients
    parabola_coeffs = np.linalg.lstsq(peak_lags, peak_values, rcond=None)[0]
    
    # Calculate the interpolated peak location (vertex of the parabola)
    interpolated_peak = -parabola_coeffs[1] / (2 * parabola_coeffs[0])
    
    return interpolated_peak


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    Estimate the time delay (tau) between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method,
    with optional parabolic interpolation.

    Parameters:
        sig: Signal (1D NumPy array).
        refsig: Reference signal (1D NumPy array).
        fs: Sampling frequency (default: 1).
        max_tau: Maximum allowable time delay (in seconds, default: None).
        interp: Interpolation factor for finer time resolution (default: 16).

    Returns:
        cross_corr: Cross-correlation values after GCC-PHAT.
        lag_values: Lag values corresponding to the cross-correlation.
    '''
    # FFT length
    n = sig.shape[0] + refsig.shape[0]
    
    # Compute cross-power spectrum
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    
    # Perform GCC-PHAT
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    
    # Define maximum lag range
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
    
    # Wrap around to limit the cross-correlation range
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    # Create lag values
    lags = np.linspace(-max_shift, max_shift, num=cc.shape[0]) / float(interp * fs)
    
    # Find the peak index
    peak_index = np.argmax(cc)
    
    # Perform parabolic interpolation
    if 0 < peak_index < len(cc) - 1:
        refined_tau = parabolic_interpolation(lags, cc)
    else:
        refined_tau = lags[peak_index]  # If peak is at edges, no interpolation
    
    return cc, lags, refined_tau



def compute_all_gcc_phat(signals, fs=1, max_tau=None, interp=16, debug_level=2):
    '''
    Compute GCC-PHAT time delays for all unique microphone pairs with parabolic interpolation.

    Parameters:
    signals: Tensor of signals from all microphones (shape: [batch_size, n_mics, signal_length]).
    fs: Sampling frequency.
    max_tau: Maximum allowable time delay (in seconds).
    interp: Interpolation factor for finer time resolution.

    Returns:
    delays: Dictionary with microphone pairs as keys and time delays (tau) as values.
    '''
    signals = signals.cpu()
    n_mics = signals.shape[1]
    mic_pairs = list(combinations(range(n_mics), 2))  # All unique pairs
    delays = {}

    for (i, j) in mic_pairs:
        # Compute GCC-PHAT cross-correlation and associated lag values
        cc, lags, refined_tau = gcc_phat(signals[0, i, :], signals[0, j, :], fs=fs, max_tau=max_tau, interp=1)
        
        if debug_level == 1:
            cc_peak_idx = np.argmax(cc)  # Index of the peak
            zoom_range = 50  # Number of samples to zoom around the peak
            
            start_idx = max(0, cc_peak_idx - zoom_range)
            end_idx = min(len(cc), cc_peak_idx + zoom_range)
            plt.figure(); plt.plot(range(max(0, start_idx), min(len(cc), end_idx)), cc[max(0, cc_peak_idx-zoom_range):min(len(cc), cc_peak_idx+zoom_range)]); plt.title('Zoomed Cross-Correlation'); plt.xlabel('Lag'); plt.ylabel('CC Amplitude'); plt.savefig('zoomed_cc.png')

            
            plt.figure(); plt.plot(cc); plt.title(f'Cross-Correlation of pair {i, j}'); plt.xlabel('Lag Index'); plt.ylabel('Amplitude'); plt.savefig('cross_correlation.png')
            plt.figure(); plt.plot(cc); plt.title(f'Cross-Correlation Zoom'); plt.xlabel('Lag Index'); plt.ylabel('Amplitude'); plt.savefig('cross_correlation.png')
            # plt.figure(); plt.plot(signals[0, i, :]); plt.title('signal1'); plt.xlabel('time'); plt.ylabel('Amplitude'); plt.savefig('signal_1.png')
            # plt.figure(); plt.plot(signals[0, j, :]); plt.title('signal2'); plt.xlabel('time'); plt.ylabel('Amplitude'); plt.savefig('signal_2.png')
            plt.figure(); plt.plot(signals[0, i, :], label='Signal 1'); plt.plot(signals[0, j, :], label='Signal 2'); plt.title('Signal 1 and Signal 2'); plt.xlabel('Time'); plt.ylabel('Amplitude'); plt.legend(); plt.savefig('signals_plot.png')

        # Apply parabolic interpolation to refine the lag estimate
        interpolated_tau = parabolic_interpolation(lags, cc)
        
        # Store the result in the dictionary
        delays[(i, j)] = interpolated_tau

    return delays


def estimate_doa_multiple_micsV2(delays, mic_positions, sound_speed=343):
    '''
    Estimate DOA for all microphone pairs using far-field approximation.

    Parameters:
    delays: Dictionary with microphone pairs as keys and time delays (tau) as values.
    mic_positions: Array of microphone positions (shape: [n_mics, 3] for 3D or [n_mics, 2] for 2D).
    sound_speed: Speed of sound in m/s (default: 343 m/s).

    Returns:
    doa_per_pair: A dictionary with microphone pairs as keys and their DOA (angle in radians) as values.
    '''
    mic_positions = mic_positions.cpu()
    doa_per_pair = {}

    mic_pairs = list(delays.keys())
    for (i, j) in mic_pairs:
        tau = delays[(i, j)]
        delta_pos = mic_positions[0,j,:] - mic_positions[0,i,:]
        norm_delta_pos = norm(delta_pos)
        
        # Calculate the cosine of the angle of arrival
        cos_theta = sound_speed * tau / norm_delta_pos
        
        # Clamp the value to avoid numerical errors (cosine should be in [-1, 1])
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # Calculate the angle using arccos
        angle = np.arccos(cos_theta)
        
        # Store the angle for this pair
        doa_per_pair[(i, j)] = np.degrees(angle)

    return doa_per_pair

def estimate_doa_multiple_mics(delays, mic_positions, sound_speed=343):
    '''
    Estimate DOA using far-field approximation from time delays and microphone positions.

    Parameters:
    delays: Dictionary with microphone pairs as keys and time delays (tau) as values.
    mic_positions: Array of microphone positions (shape: [n_mics, 3] for 3D or [n_mics, 2] for 2D).
    sound_speed: Speed of sound in m/s (default: 343 m/s).

    Returns:
    doa: Estimated direction of arrival (unit vector for 3D or angle for 2D).
    '''
    # Initialize a matrix for solving the direction vector
    A = []
    b = []

    mic_pairs = list(delays.keys())
    for (i, j) in mic_pairs:
        tau = delays[(i, j)]
        delta_pos = mic_positions[j] - mic_positions[i]
        norm_delta_pos = norm(delta_pos)
        A.append(delta_pos / norm_delta_pos)
        b.append(sound_speed * tau / norm_delta_pos)

    A = np.array(A)
    b = np.array(b)

    # Solve for the direction vector using least squares
    doa, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    doa = doa / norm(doa)  # Normalize to get a unit vector

    return doa


def calculate_true_delays(mic_positions, true_doa, sound_speed=343):
    '''
    Calculate expected time delays for a given true DOA and microphone positions.

    Parameters:
    mic_positions: Array of microphone positions (shape: [n_mics, 3] for 3D or [n_mics, 2] for 2D).
    true_doa: True direction of arrival (unit vector for 3D or angle for 2D).
    sound_speed: Speed of sound in m/s (default: 343 m/s).

    Returns:
    true_delays: Dictionary with microphone pairs as keys and expected time delays (tau) as values.
    '''
    mic_pairs = list(combinations(range(len(mic_positions)), 2))
    true_delays = {}
    
    # Convert 2D angle to a unit vector if necessary
    if len(true_doa) == 1:  # 2D case
        true_doa = np.array([np.cos(true_doa[0]), np.sin(true_doa[0])])
    
    for (i, j) in mic_pairs:
        delta_pos = mic_positions[j] - mic_positions[i]
        projected_dist = np.dot(delta_pos, true_doa)
        tau = projected_dist / sound_speed
        true_delays[(i, j)] = tau
    
    return true_delays


def simulate_signals(mic_positions, true_doa, sound_speed, fs, signal_length=1000, frequency=440):
    """
    Simulate microphone signals for a source arriving at a given DOA.

    Parameters:
    mic_positions: Array of microphone positions (shape: [n_mics, 2] for 2D or [n_mics, 3] for 3D).
    true_doa: True DOA in degrees.
    sound_speed: Speed of sound in m/s.
    fs: Sampling frequency.
    signal_length: Number of samples in the signal.
    frequency: Frequency of the simulated sine wave.

    Returns:
    signals: List of simulated signals for each microphone.
    """
    # Compute the time delays for the given true DOA
    angle_rad = np.radians(true_doa)
    direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])  # Unit vector in DOA direction
    distances = mic_positions @ direction_vector
    time_delays = distances / sound_speed

    # Generate signals
    signals = [
        np.sin(2 * np.pi * frequency * np.linspace(0, signal_length / fs, signal_length) - delay * fs)
        for delay in time_delays
    ]
    return signals

def main():
    fs = 16000
    sound_speed = 343  # Speed of sound in m/s

    # True DOA (ground truth)
    true_doa = 42  # degrees

    # Microphone positions
    mic_positions = np.array([
        [0, 0.1],       # Mic 1
        [0, 0.2],     # Mic 2
        [0, 0.3], # Mic 3
        [0, 0.4],     # Mic 4
    ])

    # Simulate signals for the true DOA
    # distance = mic_positions[:, 0] * np.cos(np.radians(true_doa)) + \
    #            mic_positions[:, 1] * np.sin(np.radians(true_doa))
    # time_delays = distance / sound_speed
    # signals = [
    #     np.sin(2 * np.pi * 440 * np.linspace(0, 1, fs) - delay * fs)
    #     for delay in time_delays
    # ]
    
    signals = simulate_signals(mic_positions, true_doa=true_doa, sound_speed=sound_speed, fs=fs)


    # Compute time delays for all unique microphone pairs
    delays = compute_all_gcc_phat(signals, fs=fs)

    # Estimate DOA based on delays and microphone geometry
    doaV2 = estimate_doa_multiple_micsV2(delays, mic_positions, sound_speed)
    doa = estimate_doa_multiple_mics(delays, mic_positions, sound_speed)

    # Output
    if mic_positions.shape[1] == 2:
        # For 2D arrays, return the angle in degrees
        angle = np.arctan2(doa[1], doa[0]) * 180 / np.pi
        print(f"True DOA: {true_doa:.2f} degrees")
        print(f"Estimated DOA: {angle:.2f} degrees")
    else:
        # For 3D arrays, return the direction vector
        print(f"Estimated DOA: {doa}")

if __name__ == "__main__":
    main()
