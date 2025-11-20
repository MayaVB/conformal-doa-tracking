import h5py
import os
import soundfile as sf
import numpy as np
from scipy.signal import lfilter

def interpolate_angles(start_deg, end_deg, num):
    """Interpolate angles correctly over circular range [0, 360)."""
    delta = (end_deg - start_deg + 180) % 360 - 180
    return (start_deg + np.linspace(0, 1, num) * delta) % 360


def get_trj_DOA(scene, signals):
    """ calculate DOA azimuth for every point in trajectory moving speaker case"""
    src_pos_trj = np.array(scene['src_pos'])  # shape [3, N]
    n_rirs = src_pos_trj.shape[1]
    n_samples = signals.shape[1]
    az_DOA = scene['DOA_az']  # shape: [n_rirs]

    doa_trj = np.zeros(n_samples)

    for i in range(n_rirs - 1):
        start = i * n_samples // n_rirs
        end = (i + 1) * n_samples // n_rirs
        doa_trj[start:end] = interpolate_angles(az_DOA[i], az_DOA[i + 1], end - start)

    # Final segment
    last_start = (n_rirs - 1) * n_samples // n_rirs
    doa_trj[last_start:] = az_DOA[-1]

    return(doa_trj)

def save_data_h5py(rev_signals, scene, scene_idx, src_pos_index, speaker_out_dir=[]):
    """
    Save wav data and label DOA in two groups, this is done for each scene and for each source position
    :param rev_signals: reverberent signals
    :param scene: scene parameters
    :param scene_idx: scene index
    :param src_pos_index: source position index
    :param speaker_out_dir: directory where the dataset is stored (defults is home dir)
    """
    # os.makedirs(speaker_out_dir, exist_ok=True)
    dataset_path = os.path.join(speaker_out_dir, 'RevMovingSrcDataset.h5')

    with h5py.File(dataset_path, 'a') as h5f:
        group_name = f"sceneIndx_{scene_idx}_roomDim_{scene['room_dim']}_rt60_{scene['RT60']}_srcPosInd_{src_pos_index}"
        grp = h5f.create_group(group_name)
        grp.create_dataset(f'signals_', data=rev_signals)

        # For trajectory scenarios, leave speaker_DOA_ empty since DOA changes over time
        if 'DOA_az_trj' in scene:
            grp.create_dataset(f'speaker_DOA_', data=np.array([]))  # Empty array for trajectory
            grp.create_dataset(f'speaker_DOA_trj', data=scene['DOA_az_trj'])
        else:
            grp.create_dataset(f'speaker_DOA_', data=scene['DOA_az'][src_pos_index])

        grp.create_dataset(f'reverberation_time_', data=scene['RT60'])
        grp.create_dataset(f'mic_positions', data=scene['mic_pos'])

    # Increment the dataset length
        if 'data_len' in h5f.attrs:
            h5f.attrs['data_len'] += 1
        else:
            h5f.attrs['data_len'] = 1
        

def save_wavs(rev_signals, file, speaker_out_dir, fs):
    for j, sig in enumerate(rev_signals, 1):
        out_file_name = '_ch{}'.format(j) + '.wav' 
        out_file_path = os.path.join(speaker_out_dir, out_file_name)
        sf.write(out_file_path, sig, fs)


def round_numbers(list_in):
    return list(map(lambda x: round(x, 4), list_in))


def calculate_speaker_speed(scene, sentence_duration):
    """
    Calculate speaker speed from trajectory positions and sentence duration
    :param scene: scene parameters containing src_pos trajectory
    :param sentence_duration: duration of sentence in seconds
    :return: speed in m/s
    """
    src_pos = scene['src_pos']  # [3, N] array of positions
    if src_pos.shape[1] < 2:
        return 0.0  # No movement for single position

    # Calculate total distance along trajectory
    total_distance = 0.0
    for i in range(src_pos.shape[1] - 1):
        pos1 = src_pos[:, i]
        pos2 = src_pos[:, i + 1]
        distance = np.linalg.norm(pos2 - pos1)
        total_distance += distance

    # Speed = total distance / time
    speed = total_distance / sentence_duration if sentence_duration > 0 else 0.0
    return speed


def write_scene_to_file(scenes, file_name, sentence_duration=None):
    """
    print scene info to a txt file
    :param scenes: list of dict containing info about the scenario generated
    :param file_name: file name to save the info
    :param sentence_duration: optional sentence duration in seconds for speed calculation
    """
    with open (file_name, 'w') as f:
        for scene in scenes:
            f.write(f"Room size: {round_numbers(scene['room_dim'])}\n")
            f.write(f"Reverberation time: {(scene['RT60'])}\n")
            f.write(f"Critical distance: {round(scene['critic_dist'], 4)}\n")
            pos = round_numbers(scene['src_pos'][0])
            f.write(f"Source position: {pos}\n")

            # Add speaker speed if sentence duration is provided
            if sentence_duration is not None:
                speed = calculate_speaker_speed(scene, sentence_duration)
                f.write(f"Speaker speed: {round(speed, 4)} m/s\n")

            mics_num = len(scene['mic_pos'])
            for i in range(mics_num):
                pos = round_numbers(scene['mic_pos'][i])
                dist = round_numbers(scene['dists'][:, i])
                f.write(f"Mic{i} pos\t: {pos}, dist:{dist}\n")
            f.write('\n\n\n')
            f.flush()
    return


def read_hdf5_file(file_path):
    data_dict = {}
    
    with h5py.File(file_path, 'r') as h5f:
        # List all groups
        group_keys = list(h5f.keys())
        print("Groups in HDF5 file:", group_keys)
        
        for group in group_keys:
            print(f"Group: {group}")
            data_dict[group] = {}
            grp = h5f[group]
            
            dataset_keys = list(grp.keys())
            print(f"Datasets in {group}:", dataset_keys)
            
            for dataset in dataset_keys:
                data = grp[dataset][:]
                print(f"Dataset: {dataset}, Data shape: {data.shape}")
                data_dict[group][dataset] = data
                
                # Example: print the first few samples
                print(f"First few samples of {dataset}: {data[:10]}")
    
    return data_dict




def generate_pink_noise(N):
    """
    Generate pink noise using the Voss-McCartney algorithm approximation.
    """
    num_rows = 16
    array = np.random.randn(num_rows, N)
    array = np.cumsum(array, axis=1)
    pink = np.sum(array, axis=0)
    pink -= np.mean(pink)
    pink /= np.std(pink)
    return pink


def inject_pink_noise_bursts(signal, fs, burst_duration=0.2, snr_db=-5, num_bursts=2):
    """
    Inject bursts of pink noise into the signal to degrade quality.

    Args:
        signal (np.ndarray): [mics, samples]
        fs (int): sample rate
        burst_duration (float): duration of each burst in seconds
        snr_db (float): SNR of burst relative to original signal
        num_bursts (int): number of bursts to inject

    Returns:
        signal_noisy (np.ndarray): same shape, with bursts added
    """
    signal_noisy = signal.copy()
    num_samples = signal.shape[1]
    burst_len = int(burst_duration * fs)

    for _ in range(num_bursts):
        start = np.random.randint(0, num_samples - burst_len)
        pink = generate_pink_noise(burst_len)

        for m in range(signal.shape[0]):
            target = signal[m, start:start+burst_len]
            signal_power = np.mean(target ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power) * pink
            signal_noisy[m, start:start+burst_len] += noise

    return signal_noisy
