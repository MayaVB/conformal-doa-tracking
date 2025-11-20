import numpy as np
import matplotlib.pyplot as plt

from torch.fft import ifft, ifftshift
from scipy.linalg import cholesky


def safe_cholesky(matrix, epsilon=1e-10):
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        # print("Cholesky decomposition failed. Using identity matrix.")
        return np.eye(matrix.shape[0])  # Return identity matrix


def EVG_RTF_estimation(signals_stft, fs, ref_channel, overlap, frame_size, noise_tim_st=0, noise_tim_fn=0.2, use_noise_esti=False, samples4RTF_esti=4096, debug_level=1):
    # Calculate new samples per frame
    R = int(frame_size * (1 - overlap))
    
    # Initial frame for RTF estimation
    frames4RTF_esti = int(samples4RTF_esti // R)
    first_frm_st = int(np.ceil(noise_tim_fn * fs / R + 1))
    
    # STFT signal setup
    z_k = signals_stft.transpose(2, 0, 1)  # shape (K, M, L) (mics, freqs/2, frames)
    M, K, L = z_k.shape
    
    # Noise processing: calculate noise correlation and Cholesky decomposition
    noise_frm_st = int(np.ceil(noise_tim_st * fs / R + 1))
    noise_frm_fn = int(np.floor(noise_tim_fn * fs / R))
    z_n = z_k[:, :, noise_frm_st:noise_frm_fn]
    
    epsilon = 1e-10
    noise_cor = np.zeros((K, M, M), dtype=np.complex128)
    noise_cor_chol = np.zeros((K, M, M), dtype=np.complex128)
    inv_chol = np.zeros((K, M, M), dtype=np.complex128)
    
    # Compute noise correlation and Cholesky once
    if use_noise_esti:
        for k in range(K):
            temp_noise = z_n[:, k, :].reshape(M, -1)
            noise_cor[k] = np.dot(temp_noise, temp_noise.T) / z_n.shape[2]
            noise_cor_chol[k] = cholesky(noise_cor[k] + epsilon * np.eye(M), lower=True)
            inv_chol[k] = np.linalg.inv(noise_cor_chol[k])
    else:
        inv_chol[:] = np.eye(M) + epsilon
        noise_cor_chol[:] = np.eye(M) + epsilon
    
    # Initialize G_f_full array to store all frames
    G_f_full_stacked = []
    
    # Process frames with a sliding window
    for frame_idx in range(first_frm_st, L - frames4RTF_esti + 10):
        frm_fn = frame_idx + frames4RTF_esti
        z_f = z_k[:, :, frame_idx:frm_fn]  # shape (K, M, frames4RTF_esti)
        
        # Precompute constants for each frame
        G_f = np.zeros((K, M), dtype=np.complex128)
        
        for k in range(K):
            temp_first = z_f[:, k, :].reshape(M, -1)  # shape (M, frames4RTF_esti)
            temp_first = np.linalg.solve(inv_chol[k], temp_first)  # solve in-place
            
            z_f_cor = np.dot(temp_first, temp_first.T) / temp_first.shape[1]
            v, w = np.linalg.eig(z_f_cor)
            fi = w[:, np.argmax(v)]  # Get eigenvector corresponding to largest eigenvalue
            
            G_f[k] = np.dot(noise_cor_chol[k], fi) / fi[ref_channel]
        
        # # Filter outliers
        # G_f[np.abs(G_f) > 3 * np.mean(np.abs(G_f), axis=0)] = 2 * np.random.binomial(1, 0.5, G_f.shape[0]) - 1
        
        # Create full G_f for the current window
        G_f_full = np.vstack([G_f, np.conj(G_f[-2::-1])])  # Reflect the array
        G_f_full = np.delete(G_f_full, ref_channel, axis=-1)
        G_f_full_stacked.append(G_f_full)
        
        if frame_idx % 30 == 0 and debug_level == 2:
            # Inverse FFT for current window and plot
            g_f = np.fft.ifft(G_f_full, axis=0)
            plt.figure()
            plt.plot(np.fft.ifftshift(g_f))
            plt.title(f'EVG_RTF (Frame {frame_idx})')
            plt.xlabel('Frequency Bins')
            plt.ylabel('Magnitude')
            plt.grid()
            plt.savefig(f'dataset_verification/EVG_RTF_frame_{frame_idx}.jpeg', format='jpeg', dpi=300)
            plt.close()
    
    # Stack the results for all frames
    G_f_full_final = np.stack(G_f_full_stacked, axis=-1)  # shape (M, frames, num_stacked_frames)
    
    Kf, T, M1 = G_f_full_final.shape
    ranks = np.zeros((Kf, T), dtype=int)

    # numerical threshold
    # scale by max singular value for robustness
    for k in range(Kf):
        for t in range(T):
            h = G_f_full_final[k, t, :]          # shape [M_minus_1]
            H = np.outer(h, h.conj())    # (M1 x M1)
            s = np.linalg.svd(H, compute_uv=False)
            tol = 10 * np.finfo(float).eps * (s.max() if s.size else 1.0)
            ranks[k, t] = int((s > tol).sum())
            
            eigvals = np.linalg.eigvalsh(H)   # automatically sorted ascending
            eigvals_sorted = np.sort(np.real(eigvals))[::-1]  # descending order

            print("Top 10 eigenvalues:")
            print(eigvals_sorted[:10])
    
    
    # Reorder and return final result
    return G_f_full_final.transpose(0, 2, 1)  # Final shape: (M, num_stacked_frames, frames)


def filter_VAD_in_stft_domain(signals_stft, vad_binary, frame_size, overlap, fs):
    # Get dimensions
    freq, frames, mics = signals_stft.shape
    time_samples = vad_binary.shape[0]

    # Calculate the step size between frames based on overlap
    R = int(frame_size * (1 - overlap))  # Step size in terms of samples between frames

    # Initialize vad_binary_pic with the same shape as signals_stft (frames, mics)
    vad_binary_framed = np.ones((frames,), dtype=bool)

    # Loop through each frame and microphone
    for frame in range(frames):
        # Calculate the start and end sample indices for this frame
        start_sample = frame * R
        end_sample = min(start_sample + frame_size, time_samples)  # Ensure we don't exceed time_samples
        
        # Check if all time samples in the frame are True- else VAD is False
        if not np.all(vad_binary[start_sample:end_sample]):
            vad_binary_framed[frame] = False

    signals_stft_filtered = signals_stft[:, vad_binary_framed, :]

    return signals_stft_filtered, vad_binary_framed 
