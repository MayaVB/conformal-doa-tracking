import gpuRIR
import torch

def generate_rirs(room_sz, all_pos_src, index, pos_rcv, T60, fs, simulate_trj = False):
    """
    Generates Room Impulse Responses (RIRs) for multiple source positions.
    
    :param room_sz: Dimensions of the room [length, width, height]
    :param pos_src: Array of source positions [[x1, y1, z1], [x2, y2, z2], ...] [mics, 3=(x,y,z)]
    :param pos_rcv: Array of receiver positions [[x1, y1, z1], [x2, y2, z2], ...]
    :param T60: Reverberation time
    :param fs: Sampling frequency
    :return: RIRs for each source position
    """
    mic_pattern = "omni"  # Receiver polar pattern
    beta = gpuRIR.beta_SabineEstimation(room_sz, T60)  # Reflection coefficients
    Tmax = T60 * 0.8  # Time to stop the simulation [s]
    Tdiff = 0.1  # ISM stops here and the Gaussian noise with an exponential envelope starts
    nb_img = gpuRIR.t2n(T60, room_sz)  # Number of image sources in each dimension
    
    if simulate_trj:
        pos_src = all_pos_src.T  # Reshape to (N, 3)
    else:
        pos_src = all_pos_src[:, index]
        pos_src = pos_src.reshape(-1, 3)  # Reshape to (N, 3) (no traj- (N=1,3))
    
    # pos_rcv_tst = np.array([[1, 1, 1], [1, 1, 1.5], [1, 1.5, 1], [2, 1.5, 1], [2.5, 1.5, 1]]) 
    # pos_rcv_tst = np.array([[1, 1, 1], [1, 1, 1.5], [1, 1.5, 1]]) 
    # pos_src_tst = np.array([[1,2.9,0.5],[1,2,0.5]]) # Positions of the 2 sources ([m]

    torch.cuda.empty_cache()  # Frees unused memory
    torch.cuda.ipc_collect()  # Helps with deallocating cache

    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, mic_pattern=mic_pattern)

    return RIRs

