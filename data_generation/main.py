from scipy.signal import lfilter
import soundfile as sf
import numpy as np

import argparse
import os
import random

from scene_gen import select_random_speaker, generate_scenes, plot_scene_interactive, calculate_doa

from utils import save_wavs, write_scene_to_file, save_data_h5py
from utils import get_trj_DOA, interpolate_angles, inject_pink_noise_bursts

from processing import add_ar1_noise_to_signals
# import matplotlib.pyplot as plt
from rir_gen import generate_rirs

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "signal_generator_python_final"))
from signal_generator import SignalGenerator

def save_args_to_file(args, filepath):
    with open(filepath, "a") as f:   # "a" = append so scene info still goes in
        f.write("\n\n===== Command-line Arguments =====\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

def _sample_noise_source_position(scene, args):
    """
    Pick a single static noise source in cylindrical coords around the array center.
    Returns a 3-vector [x, y, z].
    """
    mic_pos = np.asarray(scene['mic_pos'])         # [M,3]
    array_center = mic_pos.mean(axis=0)
    r_min = getattr(args, 'source_min_radius', 1.5)
    r_max = getattr(args, 'source_max_radius', 1.7)
    z_min = getattr(args, 'source_min_height', 1.6)
    z_max = getattr(args, 'source_max_height', 1.8)

    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(np.radians(30), np.radians(150))
    z = np.random.uniform(z_min, z_max)

    x = array_center[0] + r * np.cos(theta)
    y = array_center[1] + r * np.sin(theta)
    # clamp inside room margins
    Lx, Ly, Lz = scene['room_dim']
    margin = getattr(args, 'margin', 0.5)
    x = float(np.clip(x, margin, Lx - margin))
    y = float(np.clip(y, margin, Ly - margin))
    z = float(np.clip(z, margin, Lz - margin))
    
    # if scene['src_pos'].shape[1] < 32:
    # # 25 classes- DOAs 30,150
    #     DOA_class_selected = random.randint(0, 24)
    #     x = scene['src_pos'][0,DOA_class_selected]
    #     y = scene['src_pos'][1,DOA_class_selected]
    #     z = scene['src_pos'][2,DOA_class_selected]
    # else:
    #     # 32 classes- DOAs 10,165
    #     DOA_class_selected = random.randint(4, 31)
    #     x = scene['src_pos'][0,DOA_class_selected]
    #     y = scene['src_pos'][1,DOA_class_selected]
    #     z = scene['src_pos'][2,DOA_class_selected]
        
    return np.array([x, y, z], dtype=np.float32)

def _pink_noise(n, rng=np.random):
    """
    Approximate pink noise (1/f) via FFT shaping of white noise.
    Uses 1/sqrt(k) amplitude tilt in the frequency domain (k = bin index).
    """
    x = rng.standard_normal(n).astype(np.float64)
    X = np.fft.rfft(x)

    k = np.arange(X.size, dtype=np.float64)
    k[0] = 1.0  # avoid div-by-zero at DC
    tilt = 1.0 / np.sqrt(k)

    X *= tilt
    y = np.fft.irfft(X, n=n).astype(np.float64)

    # unit RMS
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    return (y / rms).astype(np.float32)


def _make_noise_burst(n_samples, fs, color="white"):
    """
    Make a short noise burst (white or pink) with 10 ms cosine fades and unit RMS.
    """
    if color == "pink":
        burst = _pink_noise(n_samples)
    else:
        burst = np.random.randn(n_samples).astype(np.float32)

    # 10 ms cosine fade in/out
    fade_len = max(1, int(0.01 * fs))
    win = np.ones(n_samples, dtype=np.float32)
    t = np.linspace(0, np.pi, fade_len, endpoint=True, dtype=np.float32)
    win[:fade_len] *= 0.5 * (1 - np.cos(t))
    win[-fade_len:] *= 0.5 * (1 - np.cos(t[::-1]))
    burst *= win

    # unit RMS (re-normalize after windowing)
    rms = np.sqrt(np.mean(burst**2)) + 1e-12
    return burst / rms


def inject_directional_noise_bursts(
    rev_signals, fs, scene,
    duration_range=(0.3, 0.5),
    num_bursts=3,
    snr_db=0.0,
    max_tries=50,
    color="white",        # <— new
):

    """
    Add num_bursts directional noise bursts by:
      1) Picking a random static noise source in the room;
      2) Generating RIRs from that point to all mics;
      3) Convolving a short noise burst with those RIRs; and
      4) Mixing at target SNR into a random time window.

    rev_signals: [M, T] float32
    Returns modified rev_signals.
    """
    M, T = rev_signals.shape

    # from rir_gen import generate_rirs  # reuse your existing RIR generator

    for _ in range(num_bursts):
        # burst length
        dur = float(np.random.uniform(duration_range[0], duration_range[1]))
        L = int(max(1, round(dur * fs)))
        if L >= T:
            continue

        # choose start so the whole burst fits
        tries = 0
        start = None
        while tries < max_tries:
            candidate = np.random.randint(0, T - L)
            # (optional) avoid overlapping previous loud regions; we’ll keep it simple
            start = candidate
            break
        if start is None:
            continue

        # pick a directional point & make per-mic burst via RIRs
        src = _sample_noise_source_position(scene, args=None)  # args not needed here
        # generate RIRs expects src_pos like scene['src_pos'][:, index]
        tmp_src_pos = np.stack([src], axis=1)  # [3,1]
        RIRs = generate_rirs(
            room_sz=scene['room_dim'],
            all_pos_src=tmp_src_pos,
            index=0,
            pos_rcv=scene['mic_pos'],
            T60=scene['RT60'],
            fs=fs,
            simulate_trj=False
        )
        # Generate burst exactly the needed length L
        burst_mono = _make_noise_burst(L, fs, color=color)

        # Convolve into each mic channel
        # RIRs[0] is list of length M, each: rir[k]
        burst_mics = np.zeros((M, L), dtype=np.float32)
        for m, rir in enumerate(RIRs[0]):
            y = lfilter(rir.astype(np.float32), 1.0, burst_mono)
            # Truncate to exactly L samples (no padding needed since input is L)
            y = y[:L]
            # Normalize per-mic so combined RMS is stable before SNR scaling
            burst_mics[m] = y

        # Compute gain for target SNR (use all mics jointly over the window)
        seg = rev_signals[:, start:start+L]
        sig_rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
        noise_rms = float(np.sqrt(np.mean(burst_mics**2)) + 1e-12)
        # snr_db = 20*log10(sig_rms / noise_rms_scaled)  -> gain = sig_rms / (noise_rms * 10^(snr/20))
        gain = sig_rms / (noise_rms * (10.0**(snr_db/20.0)))
        seg += gain * burst_mics
        rev_signals[:, start:start+L] = seg

    return rev_signals


def build_three_phase_envelope(n_samples, fs, quiet_start_s, quiet_duration_s, ramp_s, quiet_gain):
    """
    3 phases: normal(1.0) -> quiet(quiet_gain) -> back to normal(1.0), with cosine ramps.
    Times are clamped to the signal length.
    """
    import numpy as np
    env = np.ones(n_samples, dtype=np.float32)

    # times -> samples
    qs = int(max(0, quiet_start_s) * fs)
    qd = int(max(0, quiet_duration_s) * fs)
    re = int(max(0, ramp_s) * fs)

    q_start = min(qs, n_samples)
    q_end   = min(qs + qd, n_samples)

    # Early exit if quiet window degenerate
    if q_end <= q_start or quiet_gain >= 0.999:
        return env

    # Ramp into quiet
    ramp_in_end = min(q_start + re, q_end)
    if ramp_in_end > q_start:
        t = np.linspace(0, np.pi, ramp_in_end - q_start, endpoint=True)
        env[q_start:ramp_in_end] = 1.0 + (quiet_gain - 1.0) * (0.5 * (1 - np.cos(t)))  # cosine fade down

    # Steady quiet
    steady_start = ramp_in_end
    steady_end   = max(steady_start, q_end - re)
    if steady_end > steady_start:
        env[steady_start:steady_end] = quiet_gain

    # Ramp out of quiet (back to 1.0)
    ramp_out_start = steady_end
    ramp_out_end   = min(q_end, ramp_out_start + re)
    if ramp_out_end > ramp_out_start:
        t = np.linspace(0, np.pi, ramp_out_end - ramp_out_start, endpoint=True)
        env[ramp_out_start:ramp_out_end] = quiet_gain + (1.0 - quiet_gain) * (0.5 * (1 - np.cos(t)))

    # Past quiet: stays at 1.0
    return env

def build_paths_from_scene(s, fs, scene, fps):
    """
    Build source and receiver paths for trajectory simulation.
    Returns sp_path [T,3], rp_path [T,3,M] using the scene's arc (scene['src_pos']).

    Note: sp_path and rp_path are also used in matlab_sg_wrapper.py and
    signal_generator_python_final/run_example.py but serve the same purpose.
    """
    T = len(s)
    hop = max(1, int(round(fs / float(fps))))
    src_pos = np.asarray(scene['src_pos'])        # [3, N]
    mic_pos = np.asarray(scene['mic_pos'])        # [M, 3]
    M = mic_pos.shape[0]

    # Receiver path: static mics repeated over time
    rp_path = np.zeros((T, 3, M), dtype=np.float32)
    for m in range(M):
        rp_path[:, :, m] = mic_pos[m]  # broadcast constant mic position

    # Source path - walk the existing arc defined by the discrete grid src_pos[:, k]
    sp_path = np.zeros((T, 3), dtype=np.float32)
    N = src_pos.shape[1]
    if N < 2:
        sp_path[:] = src_pos[:, 0]
    else:
        # step across the arc every 'hop' samples; linear interp between arc points
        k = 0
        for i in range(0, T, hop):
            j = min(i + hop, T)
            k_next = min(k + 1, N - 1)
            # alpha inside segment based on i across total time
            seg_alpha = (i / max(1, T - 1))
            # map seg_alpha to progress along [0..N-1]
            posf = seg_alpha * (N - 1)
            k_floor = int(np.floor(posf))
            k_ceil = min(k_floor + 1, N - 1)
            w = posf - k_floor
            sp = (1 - w) * src_pos[:, k_floor] + w * src_pos[:, k_ceil]
            sp_path[i:j] = sp.astype(np.float32)
            k = k_next

    return sp_path, rp_path


def generate_rev_speech(args):
    """
    :param args: From Parser
    :return: scene_agg: scence list of dict containing info about the scenario generated
    """
    scene_agg = []
    sentence_durations = []  # Store sentence durations for speed calculation
    clean_speech_dir = args.clean_speech_dir
    num_scenes = args.num_scenes
    snr = args.snr
    # simulate_cpp_trj = args.simulate_cpp_trj
    
    if args.output_folder:
        save_rev_speech_dir = os.path.join(args.split, args.output_folder)
    else:
        save_rev_speech_dir = args.split
    
    # Generate reverberant speech files
    for scene_idx in range(num_scenes):
        selected_speaker, wav_files = select_random_speaker(clean_speech_dir)

        # Generate a scene
        scene = generate_scenes(args)[0]
        mics_num = len(scene['mic_pos'])
        
        # plot an interactive scene plot for each scenarion
        plot_scene_interactive(scene, save_rev_speech_dir, scene_idx)
                
        for index, _ in enumerate(scene['src_pos'][0]):
            curr_wav_file_path = random.choice(wav_files)

            if len(wav_files) < len(scene['src_pos'][0]):
                raise Exception("speaker wav files are lower than requested speaker location- not enough files!") 

            # check wav length
            s, fs = sf.read(curr_wav_file_path) 
            if len(s)/fs > args.minimum_sentence_len:
                wav_verified = True
            else:
                wav_verified = False

            while not wav_verified:

                # Read a clean file
                s, fs = sf.read(curr_wav_file_path)
                if len(s)/fs < args.minimum_sentence_len or len(s)/fs > args.maximum_sentence_len:
                    curr_wav_file_path = random.choice(wav_files)
                else:
                    wav_verified = True

            # Store sentence duration for speed calculation
            sentence_duration = len(s) / fs
            sentence_durations.append(sentence_duration)

            print('Processing Scene %d/%d. Speaker: %s, wav file processed: %s, wav file number: %d.' % (
                scene_idx + 1, num_scenes, selected_speaker, os.path.basename(curr_wav_file_path), index + 1))

            # Generate RIR for the current source position for all mic positons
            if args.simulate_trj:
                print(f"simulate_trj argument: {args.simulate_trj}")
                # Optionally reduce trajectory resolution for performance
                if args.fast_mode:
                    reduced_fps = min(args.fps, 50)  # Limit to 50 Hz for performance in fast mode
                    print(f"Fast mode: reducing trajectory fps from {args.fps} to {reduced_fps}")
                    order=1
                else:
                    reduced_fps = args.fps
                    order=2

                sp_path, rp_path = build_paths_from_scene(
                    s=s, fs=fs, scene=scene, fps=reduced_fps
                )

                # Room & RIR settings
                L = scene['room_dim']               # [Lx, Ly, Lz]
                beta = [float(scene['RT60'])]       # reverberation time list
                Tmax = scene['RT60'] * 0.8  # Time to stop the simulation [s] - consistent with generate_rirs
                nsamples = int(max(1, round(Tmax * fs)))

                # Convert data for SignalGenerator
                input_signal_list = list(s.astype(float))
                rp_path_list = rp_path.tolist()
                sp_path_list = sp_path.tolist()
                
                print("Running SignalGenerator...")
                gen = SignalGenerator()
                result = gen.generate(
                    input_signal=input_signal_list,
                    c=340.0,
                    fs=int(fs),
                    r_path=rp_path_list,        # [T, 3, M]
                    s_path=sp_path_list,        # [T, 3]
                    L=L,
                    beta_or_tr=beta,
                    nsamples=nsamples,
                    mtype="o",
                    order=order,
                    hp_filter=False
                )

                # Pack to our expected shape [M, T]
                rev_signals = np.asarray(result.output, dtype=np.float32)  # list[M][T]
                rev_signals = np.stack(rev_signals, axis=0)                # [M, T]

                # >>> apply speaking envelope post-RIR so it survives generator scaling <<<
                if args.simulate_quiet_profile:
                    env = build_three_phase_envelope(
                        n_samples=len(s),
                        fs=fs,
                        quiet_start_s=args.quiet_start_s,
                        quiet_duration_s=args.quiet_duration_s,
                        ramp_s=args.ramp_s,
                        quiet_gain=args.quiet_gain,
                    )
                    rev_signals = rev_signals * env[None, :]  # [M,T] *= [1,T]
                    print(f"simulated quiet profile")


                # Optional: compute DOA trajectory for bookkeeping/labels
                scene['DOA_az_trj'] = get_trj_DOA(scene, rev_signals)    # expects [M, T]
                
            else: 
                
                RIRs = generate_rirs(scene['room_dim'], scene['src_pos'], index, scene['mic_pos'], scene['RT60'], fs, args.simulate_trj)

                # Generate reverberant speech for each mic
                rev_signals = np.zeros([mics_num, len(s)])
                for j, rir in enumerate(RIRs[0]):
                    rev_signals[j] = lfilter(rir, 1, s)
            
            # Normalize
            rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  
            
            # Add noise
            if args.dataset == 'add_noise':
                rev_signals = add_ar1_noise_to_signals(rev_signals, decay=0.9, snr_db=args.snr)
                # rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1  # Normalize
                
            if args.inject_burst_noise:
                rev_signals = inject_pink_noise_bursts(
                    rev_signals,
                    fs=fs,
                    burst_duration=0.5,       # seconds
                    snr_db=-5,                # make bursts *louder* than signal
                    num_bursts=10              # 2 bursts at random times
                )
                
            if args.inject_directional_noise:
                rev_signals = inject_directional_noise_bursts(
                    rev_signals=rev_signals,
                    fs=fs,
                    scene=scene,
                    duration_range=(args.dir_noise_min_ms / 1000.0, args.dir_noise_max_ms / 1000.0),
                    num_bursts=args.dir_noise_num_bursts,
                    snr_db=args.dir_noise_snr_db,
                    color=args.dir_noise_color,   # <— here
                )
            
            # normalization- hurts the noise we add
            # rev_signals = rev_signals / np.max(np.abs(rev_signals)) / 1.1
            
            # Final peak limiter only (preserves dynamics)
            peak = np.max(np.abs(rev_signals)) + 1e-9
            if peak > 0.999:
                rev_signals = 0.999 * rev_signals / peak

            # Save data as wavs and create a directory in which wavs will be saved
            speaker_wav_dir = os.path.join(save_rev_speech_dir, 'RevMovingSrcDatasetWavs', f'scene_{scene_idx + 1}', selected_speaker, f'src_pos{index + 1}')
            os.makedirs(speaker_wav_dir, exist_ok=True)
            save_wavs(rev_signals, 0, speaker_wav_dir, fs)

            # Save data as h5py file
            save_data_h5py(rev_signals, scene, scene_idx, index, save_rev_speech_dir)
            
            if args.simulate_trj:
                break # for traj simulation we dont need to loop over src_pos the calculation is performed at "simulateTrajectory"
        
        scene_agg.append(scene)
        
        # save scene info txt file
        avg_sentence_duration = np.mean(sentence_durations) if sentence_durations else None
        write_scene_to_file(scene_agg, os.path.join(save_rev_speech_dir, 'dataset_info.txt'), avg_sentence_duration)
        save_args_to_file(args, os.path.join(save_rev_speech_dir, 'dataset_info.txt'))

    return scene_agg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a clean sound sample."
    )
    # general parameters
    parser.add_argument("--split", type=str, default="../dataset_folder/gannot-lab/gannot-lab1/SpeakerLocGen/", help="Dataset spl")
    parser.add_argument("--clean_speech_dir", type=str, default='../dataset_folder/gannot-lab/gannot-lab1/datasets/sharon_db/wsj0/Train/', help="Directory where the clean speech files are stored")
    parser.add_argument("--dataset", choices=['None', 'add_noise'], default='add_noise')
    
    # output folder
    parser.add_argument("--output_folder", type=str, default='', help="Directory where the output is saved")

    # scene parameters
    parser.add_argument("--num_scenes", type=int, default=5, help="Number of scenarios to generate")
    parser.add_argument("--mics_num", type=int, default=5, help="Number of microphones in the array")
    parser.add_argument("--mic_min_spacing", type=float, default=0.08, help="Minimum spacing between microphones")
    parser.add_argument("--mic_max_spacing", type=float, default=0.08, help="Maximum spacing between microphones")
    parser.add_argument("--mic_height", type=float, default=1.7, help="Height of the microphone array from the floor")
    parser.add_argument("--margin", type=float, default=0.5, help="Margin distance between the source/mics to the walls")  

    parser.add_argument("--room_len_x_min", type=float, default=4, help="Minimum length of the room in x-direction")
    parser.add_argument("--room_len_x_max", type=float, default=7, help="Maximum length of the room in x-direction")
    parser.add_argument("--aspect_ratio_min", type=float, default=1, help="Minimum aspect ratio of the room")
    parser.add_argument("--aspect_ratio_max", type=float, default=1.5, help="Maximum aspect ratio of the room")
    parser.add_argument("--room_len_z_min", type=float, default=2.3, help="Minimum length of the room in z-direction")
    parser.add_argument("--room_len_z_max", type=float, default=2.9, help="Maximum length of the room in z-direction")

    # parser.add_argument("--T60_options", type=float, nargs='+', default=[0.3], help="List of T60 values for the room [0.2, 0.4, 0.6, 0.8], [0.3, 0.5, 0.8]")
    parser.add_argument("--T60_options", type=float, nargs='+', choices=[0.2, 0.5, 0.8], default=[0.2], help="Choose one or more T60 values from [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]")    
    parser.add_argument("--snr", type=float, default=20, help="added noise snr value [dB]")
    parser.add_argument("--noise_fc", type=float, default=1000, help="cufoff lowpass freq for added noise [Hz]")
    parser.add_argument("--noise_AR_decay", type=float, default=0.6, help="cufoff lowpass freq for added noise [Hz]")
    parser.add_argument("--minimum_sentence_len", type=float, default=8, help="default = 8 minimm required for sentence length in seconds was 3")
    parser.add_argument("--maximum_sentence_len", type=float, default=9, help="max required for sentence length in seconds was 4")

    parser.add_argument("--source_min_height", type=float, default=1.6, help="Minimum height of the source (1.5/1.65)")
    parser.add_argument("--source_max_height", type=float, default=1.8, help="Maximum height of the source (2/1.75)")
    parser.add_argument("--source_min_radius", type=float, default=1.5, help="Minimum radius for source localization")
    parser.add_argument("--source_max_radius", type=float, default=1.7, help="Maximum radius for source localization")
    parser.add_argument("--DOA_grid_lag", type=float, default=5, help="Degrees for DOA grid lag")
    
    # Performance optimization flags and trj
    parser.add_argument("--simulate_trj", action="store_true", help="simulate moving speaker")
    parser.add_argument("--endfire_bounce", type=bool, default=False, help="simulate 'half circle' movment- endfire -> broadband - back to endfire")
    parser.add_argument("--fps", type=float, default=125, help="frames per second: fs/(framesize*(1-overlap))")
    parser.add_argument("--offgrid_angle", type=bool, default=False, help="generate off sampeling grid doas")
    parser.add_argument("--fast_mode", type=bool, default=False, help="Enable performance optimizations (reduced resolution, shorter RIRs, lower reflection order)")

    # Dual speaker scenario parameters
    parser.add_argument("--dual_speaker_opposing", type=bool, default=False, help="simulate two speakers moving in opposite directions on the same arc")
    parser.add_argument("--dual_speaker_start_angle", type=float, default=30, help="starting angle for first speaker (degrees)")
    parser.add_argument("--dual_speaker_end_angle", type=float, default=150, help="ending angle for first speaker (degrees)")
    parser.add_argument("--dual_speaker_snr_balance", type=float, default=0.0, help="SNR difference between speakers in dB (positive = speaker 1 louder)")
    
    # single_speaker_30_to_150 deg movement
    parser.add_argument("--single_speaker_30_to_150", type=bool, default=False, help="single_speaker_30_to_150 (default: True)")

    # simulate_quiet_profile
    parser.add_argument("--simulate_quiet_profile", action="store_true", help="add burst noise")
    parser.add_argument("--quiet_start_s", type=float, default=1, help="Time (s) when quiet phase begins")
    parser.add_argument("--quiet_duration_s", type=float, default=2.0, help="Length (s) of the quiet phase (middle section)")
    parser.add_argument("--ramp_s", type=float, default=0.8, help="Fade time (s) for smooth transitions in/out")
    parser.add_argument("--quiet_gain", type=float, default=0.02, help="Gain during quiet phase (0-1)")

    # Directional noise bursts (static, RIR-based)
    parser.add_argument("--inject_burst_noise", type=bool, default=False, help="add burst noise (default: True)")

    # Directional noise bursts (static, RIR-based)
    parser.add_argument("--inject_directional_noise", action="store_true", help="Add short directional noise bursts convolved with RIRs")
    parser.add_argument("--dir_noise_color", choices=["white", "pink"], default="pink", help="Spectrum of directional burst noise.")
    parser.add_argument("--dir_noise_num_bursts", type=int, default=5, help="How many directional bursts to inject")
    parser.add_argument("--dir_noise_min_ms", type=int, default=300, help="Min burst duration in milliseconds")
    parser.add_argument("--dir_noise_max_ms", type=int, default=500, help="Max burst duration in milliseconds")
    parser.add_argument("--dir_noise_snr_db", type=float, default=0.0, help="Target SNR (dB) of noise vs signal over the burst window; 0 = equal power, negative = louder noise")

    args = parser.parse_args()
    
    
    ## ======= Main: Generate data for training/validation ======= ##
    scenes = generate_rev_speech(args)
        
