import os
import random

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

epsilon = 1e-5  # A small constant

def estimate_max_order(room_sz, T60, c=343.0):
    Tmax = 0.8 * T60
    avg_dist = np.mean(room_sz)
    return int(np.ceil((c * Tmax) / avg_dist))

def center_to_corner(pos, room_sz):
    """
    Convert positions from centered coordinate system to corner-origin.
    """
    return pos + np.array(room_sz) / 2



def select_random_speaker(speakers_dir_clean):
    """
    Selects a random speaker directory and returns the speaker name and all .wav files in that directory.
    :param speakers_dir_clean: Path to the directory containing speaker directories
    :return: selected_dir: Name of the selected speaker directory
             wav_files: List of all .wav files in the selected speaker directory
    """
    # List all directories in the speakers_dir_clean
    speaker_dirs = [d for d in os.listdir(speakers_dir_clean) if os.path.isdir(os.path.join(speakers_dir_clean, d))]

    # Ensure there are directories in the speakers_dir_clean
    if not speaker_dirs:
        raise ValueError("No speaker directories found in the directory.")

    # Select a random directory
    selected_dir = random.choice(speaker_dirs)
    selected_dir_path = os.path.join(speakers_dir_clean, selected_dir)

    # List all .wav files in the selected directory
    wav_files = [os.path.join(selected_dir_path, f) for f in os.listdir(selected_dir_path) if f.endswith('.wav')]

    # Ensure there are .wav files in the selected directory
    if not wav_files:
        raise ValueError("No .wav files found in the selected speaker directory.")

    return selected_dir, wav_files


def critical_distance(V, T60):
    """ critical_distance determine the distance from a sound source at which the sound field transitions from 
    being dominated by the direct sound to being dominated by the reverberant sound in a given environment
    :param V: room volume [m^3]
    :param T60: room reverberation time [s]
    :return: critical_distance: explained in the description
    """
    return 0.057*np.sqrt(V/T60)


def calculate_doa(src_pos, mic_pos):
    """
    Calculate the Direction of Arrival (DOA) from the center of the microphone array to the source positions in angles.
    :param src_pos: The positions of the sources [3, n_sources].
    :param mic_pos: A list of positions of the microphones [[x1, y1, z1], [x2, y2, z2], ...].
    :return: The DOA azimuth angles in degrees.
    """
    mic_pos = np.array(mic_pos)
    array_center = np.mean(mic_pos, axis=0)  # Calculate the center of the microphone array
    
    # Define the array vector as the vector from the first to the last microphone position
    array_vector = mic_pos[-1] - mic_pos[0]
    array_vector = array_vector / np.linalg.norm(array_vector)  # Normalize the array vector
    
    # Vector from array center to source
    doa_vector = src_pos - array_center[:, np.newaxis]  # src_pos shape of (3, n_sources), array_center shape (3,)
    
    # Normalize the DOA vector
    norms = np.linalg.norm(doa_vector, axis=0)  # Shape: (n_sources,) 
    doa_vector_normalized = doa_vector / norms  # Normalize each column

    # Calculate the DOA azimuth angle
    dot_product = np.dot(array_vector, doa_vector_normalized)
    azimuth_angles = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    # Calculate the DOA elevation angle
    elevation = np.degrees(np.arcsin(doa_vector_normalized[2, :]))
    
    # Adjust angles to meet the desired convention (90 degrees in front, 0 and 180 degrees along the array line)
    for i in range(len(azimuth_angles)):
        cross_product = np.cross(array_vector, doa_vector_normalized[:, i])
        if cross_product[2] < 0:
            azimuth_angles[i] = 360 - azimuth_angles[i]
    
    return azimuth_angles, elevation


def generate_scenes(args):
    """
    Generates random rooms with random source and microphone positions.
    
    :param args: Input arguments from a parser that contains the room and array configuration parameters.
    :return: A list of dictionaries containing source, microphone, and room information.
    """
    start_doa_grid = 10
    end_doa_grid = 170
    endfire_bounce = args.endfire_bounce                 # Number of microphones in the array

    # Microphone array parameters
    mics_num = args.mics_num                 # Number of microphones in the array
    mic_min_spacing = args.mic_min_spacing   # Minimum spacing between microphones
    mic_max_spacing = args.mic_max_spacing   # Maximum spacing between microphones
    mic_height = args.mic_height             # Fixed height of the microphones

    # Room dimensions (lengths and aspect ratios)
    room_len_x_min = args.room_len_x_min     # Minimum room length in x-direction
    room_len_x_max = args.room_len_x_max     # Maximum room length in x-direction
    aspect_ratio_min = args.aspect_ratio_min # Minimum aspect ratio (length y to length x)
    aspect_ratio_max = args.aspect_ratio_max # Maximum aspect ratio (length y to length x)
    room_len_z_min = args.room_len_z_min     # Minimum room length in z-direction (height)
    room_len_z_max = args.room_len_z_max     # Maximum room length in z-direction (height)

    # Reverberation time (T60) selection from predefined options
    T60 = random.choice(args.T60_options)   # Randomly choose T60 from available options
    
    # Source parameters
    source_min_height = args.source_min_height   # Minimum height of the source
    source_max_height = args.source_max_height   # Maximum height of the source
    source_min_radius = args.source_min_radius   # Minimum radial distance from the microphone array
    source_max_radius = args.source_max_radius   # Maximum radial distance from the microphone array
    DOA_grid_lag = args.DOA_grid_lag             # Angle step for DOA grid calculation
    offgrid_angle = args.offgrid_angle
    
    # Minimum margin to ensure the microphones and source are not too close to room walls
    margin = args.margin

    while True:
        src_mics_info = []

        # Randomly determine room dimensions
        room_len_x = np.random.uniform(room_len_x_min, room_len_x_max)
        room_len_z = np.random.uniform(room_len_z_min, room_len_z_max)
        aspect_ratio = np.random.uniform(aspect_ratio_min, aspect_ratio_max)
        room_len_y = room_len_x * aspect_ratio  # Calculate room length in y-direction using aspect ratio
        room_dim = [*np.random.permutation([room_len_x, room_len_y]), room_len_z]  # Shuffle x, y for variability

        # Generate ULA microphone positions with spacing and margin constraints
        mic_spacing = np.random.uniform(mic_min_spacing, mic_max_spacing)
        mic_pos_start = np.random.uniform(margin, room_dim[0] - margin - (mics_num - 1) * mic_spacing)
        mic_pos_y = np.random.uniform(margin, room_dim[1] - margin)
        mic_pos_z = mic_height  # Fixed height for all microphones

        mics_pos_agg = []  # Aggregate array to store positions
        mics_pos_arr = np.zeros((mics_num, 3))  # Preallocate mic position array

        # Compute microphone positions along the x-axis
        for mic in range(mics_num):
            mic_x = mic_pos_start + mic * mic_spacing
            mic_pos = [mic_x, mic_pos_y, mic_pos_z]
            mics_pos_agg.append(mic_pos)  # Append position to list
            mics_pos_arr[mic] = mic_pos   # Store position in the preallocated array

        mics_pos_arr = mics_pos_arr.T  # Transpose for easier calculations later

        # Randomize the orientation of the microphone array by applying a rotation
        rotation_angle = np.deg2rad(np.round(np.random.uniform(0, 180)))  # Random rotation angle between 0 and π
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])  # 2D rotation matrix for rotation around the z-axis

        # Rotate microphone positions and translate back to original coordinates
        mics_pos_agg = [
            rotation_matrix.dot(np.array(mic) - np.array([room_dim[0] / 2, room_dim[1] / 2, 0])) + np.array([room_dim[0] / 2, room_dim[1] / 2, 0])
            for mic in mics_pos_agg
        ]

        # Find the center of the microphone array
        mic_array_center = np.mean(mics_pos_agg, axis=0)

        # Generate source positions around the microphone array within a defined radial range
        if hasattr(args, 'single_speaker_30_to_150') and args.single_speaker_30_to_150:
            # Single speaker moving from 30 to 150 degrees
            start_angle = 30
            end_angle = 150
            angles = np.arange(start_angle, end_angle + DOA_grid_lag, DOA_grid_lag)
            src_angle = np.radians(angles + np.degrees(rotation_angle))

        elif hasattr(args, 'dual_speaker_opposing') and args.dual_speaker_opposing:
            # Create opposing arcs for two speakers
            start_angle = getattr(args, 'dual_speaker_start_angle', 30)
            end_angle = getattr(args, 'dual_speaker_end_angle', 150)

            # Speaker 1: start_angle -> end_angle using DOA_grid_lag steps
            angles_spk1 = np.arange(start_angle, end_angle + DOA_grid_lag, DOA_grid_lag)
            # Speaker 2: end_angle -> start_angle using DOA_grid_lag steps (opposite direction)
            angles_spk2 = np.arange(end_angle, start_angle - DOA_grid_lag, -DOA_grid_lag)

            # Combine both speaker trajectories
            src_angle = np.radians(np.concatenate([angles_spk1, angles_spk2]) + np.degrees(rotation_angle))

        elif endfire_bounce:
            n_points = 32

            angles_forward = np.arange(start_doa_grid, 90, DOA_grid_lag)
            angles_back = angles_forward[::-1]
            src_angle = np.radians(np.concatenate([angles_forward, angles_back]) + np.degrees(rotation_angle))
            # np.degrees(src_angle)

        elif offgrid_angle:
            n_points = 32
            src_angle = np.radians(np.arange(start_doa_grid, end_doa_grid, DOA_grid_lag) + [round(random.uniform(0, DOA_grid_lag/2), 2) for _ in range(n_points)] + np.degrees(rotation_angle))  # Calculate angles
            # np.degrees(src_angle)
        else:
            src_angle = np.radians(np.arange(start_doa_grid, end_doa_grid, DOA_grid_lag) + np.degrees(rotation_angle))  # Calculate angles

        src_radius = np.random.uniform(source_min_radius, source_max_radius, size=len(src_angle))  # Random radii

        # Convert polar coordinates (angle, radius) to Cartesian coordinates (x, y)
        src_x = np.multiply(src_radius, np.cos(src_angle)) + mic_array_center[0]
        src_y = np.multiply(src_radius, np.sin(src_angle)) + mic_array_center[1]

        # Randomly assign source heights
        src_z = np.random.uniform(source_min_height, source_max_height, size=src_x.shape)

        # Stack source positions as a (3, N) array (x, y, z)
        src_pos = np.array([src_x, src_y, src_z])

        # Verify that the source positions are within the room boundaries, considering the margin
        if np.all((margin < src_x) & (src_x < room_dim[0] - margin)) and np.all((margin < src_y) & (src_y < room_dim[1] - margin)):
            break

    # Calculate distances between the microphones and each source position
    # src_pos_expanded = src_pos.T[:, np.newaxis, :]  # Expand source position array shape: (num_sources, 1, 3)
    # mics_pos_expanded = mics_pos_arr.T[np.newaxis, :, :]  # Expand mic position array shape: (1, num_mics, 3)
    # dists = np.linalg.norm(src_pos_expanded - mics_pos_expanded, axis=2)  # Calculate Euclidean distances
    
    # replaces the above with rotation- bugfix
    mics_pos_np = np.array(mics_pos_agg)          # rotated
    src_pos_expanded  = src_pos.T[:, None, :]
    mics_pos_expanded = mics_pos_np[None, :, :]
    dists = np.linalg.norm(src_pos_expanded - mics_pos_expanded, axis=2)

    # Compute the critical distance based on room volume and T60
    critic_dist = critical_distance(np.prod(room_dim), T60)

    # Calculate Direction of Arrival (DOA) angles (azimuth and elevation) - always from geometry
    az_DOA, el_DOA = calculate_doa(src_pos, np.array(mics_pos_agg))

    # Apply specific processing based on scenario type
    if endfire_bounce:
        az_DOA = az_DOA.astype(int)  # truncate and make integer for endfire
   
    # Append all relevant information to the scene list
    src_mics_info.append({
        'room_dim': room_dim,          # Room dimensions (x, y, z)
        'src_pos': src_pos,            # Source positions (x, y, z)
        'mic_pos': np.array(mics_pos_agg),  # Microphone positions (x, y, z)
        'critic_dist': critic_dist,    # Critical distance
        'SNR': args.snr,                # snr
        'dists': dists,                # Distances between mics and sources
        'RT60': T60,                   # Reverberation time
        'DOA_az': az_DOA               # Azimuth DOA of the source
    })

    return src_mics_info
        

def plot_scene(scene, save_path, index):
    """
    Plots the room shape, microphone array and speaker
    :param scene: scene parameters
    :param save_path: saving path for figure generated
    :param index: outside loop index to support multiplay plots in folder
    # NOTE :  this plot is unclear use plot_scene_interactive
    """
    room_dim = scene['room_dim']
    src_pos = scene['src_pos']
    mic_pos = scene['mic_pos']
    DOA_az = scene['DOA_az']
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot room dimensions
    ax.scatter(room_dim[0], room_dim[1], room_dim[2], c='b', marker='s', label='Room Dimensions')

    # Plot source position
    ax.scatter(src_pos[0][0], src_pos[0][1], src_pos[0][2], c='r', marker='o', label='Source Position')

    # Plot microphone positions
    for i, mic in enumerate(mic_pos):
        ax.scatter(mic[0], mic[1], mic[2], c='g', marker='^', label=f'Microphone {i+1} Position')

    # Set limits for the axes to ensure full view
    ax.set_xlim([0, room_dim[0]])  # Replace xmin and xmax with your desired limits
    ax.set_ylim([0, room_dim[1]])  # Replace ymin and ymax with your desired limits
    ax.set_zlim([0, room_dim[2]])  # Replace zmin and zmax with your desired limits

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Room with Source and Microphones. DOA is {:.2f} degrees'.format(DOA_az))
    ax.legend()

    # Save plot as JPEG
    plt.savefig(os.path.join(save_path, f"scene_plot_{index}.JPEG"))
    plt.close()
    

def plot_scene_interactive(scene, save_path, index):
    """
    Plots the room shape, microphone array, and speaker with azimuth angles.
    This is an HTML file for interactively moving the room shape.
    :param scene: scene parameters
    :param save_path: saving path for the figure generated
    :param index: outside loop index to support multiple plots in folder
    """
    save_path = os.path.join(save_path, 'plots')
    os.makedirs(save_path, exist_ok=True)

    room_dim = scene['room_dim']
    src_pos = scene['src_pos']
    mic_pos = scene['mic_pos']
    DOA_az = scene['DOA_az']

    fig = go.Figure()

    # Plot room dimensions (corners of the room for illustration)
    fig.add_trace(go.Scatter3d(
        x=[0, room_dim[0], room_dim[0], 0, 0, 0, room_dim[0], room_dim[0], 0, 0, room_dim[0], room_dim[0]],
        y=[0, 0, room_dim[1], room_dim[1], 0, 0, 0, room_dim[1], room_dim[1], 0, 0, room_dim[1]],
        z=[0, 0, 0, 0, 0, room_dim[2], room_dim[2], room_dim[2], room_dim[2], room_dim[2], room_dim[2], room_dim[2]],
        mode='lines',
        name='Room Dimensions'
    ))

    # Plot source positions with azimuth angles
    for i in range(src_pos.shape[1]):
        fig.add_trace(go.Scatter3d(
            x=[src_pos[0, i]],
            y=[src_pos[1, i]],
            z=[src_pos[2, i]],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[f'{DOA_az[i]:.2f}°'],
            textposition='top center',
            name=f'Source Position {i+1}'
        ))

    # Plot microphone positions
    for i, mic in enumerate(mic_pos):
        fig.add_trace(go.Scatter3d(
            x=[mic[0]],
            y=[mic[1]],
            z=[mic[2]],
            mode='markers',
            marker=dict(size=5, color='green'),
            name=f'Microphone {i+1} Position'
        ))

    # Set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title=f'Room with Source and Microphones'
    )

    # Save plot as an interactive HTML file
    fig.write_html(os.path.join(save_path, f"scene_plot_{index}.html"))

