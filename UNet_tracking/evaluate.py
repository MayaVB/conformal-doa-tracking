import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils.metrics import one_hot_to_angle
from utils.gcc_phat import compute_all_gcc_phat, estimate_doa_multiple_micsV2


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            x_train = batch['input']
            target = batch['target']

            x_train = x_train.to(device=device, dtype=torch.float32)
            # target_type = torch.float32 if net.n_classes == 1 else torch.long
            target_type = torch.float32
            target = target.to(device=device, dtype=target_type)
            target = torch.mean(target, dim=2)  # Average over the freqs dimension
            
            pred = net(x_train)
            if pred.shape[2] != 1:
                # if processing all freqs
                pred = torch.mean(pred, dim=2)
            else:
                pred = pred.squeeze(2)

            
            for target, pred in zip(target, pred):
                pred = (pred > 0.5).float()
                tot += F.cross_entropy(pred.unsqueeze(dim=0), target.unsqueeze(dim=0)).item()
                
            pbar.update(x_train.shape[0])

    return tot / n_val


# def evaluate_results(result, belief_over_time):
    # softmax_probs= np.stack(result['probabilities'], axis=0), 
    # CP_sets = result['CP_sets']
    
    
def evaluate_results(result, angle_step=5, start_angle=10):
    """_summary_

    Args:
        result (_type_): _description_
        belief_over_time (_type_): _description_
        angle_step (int, optional): _description_. Defaults to 5.
        start_angle (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    belief_over_time = result['belief_over_time']
    
    softmax_probs = np.stack(result['probabilities'], axis=0), 
    CP_sets = result['CP_sets']

    model_MAE = np.mean(np.abs(np.stack(result['unet_doa']) - np.stack((result['raw_doa']).cpu())))
    gcc_MAE = np.mean(np.abs(result['gcc_doa'] - np.stack((result['raw_doa']).cpu())))
    filter_MAE = np.mean(np.abs(result['filter_doa'] - np.stack((result['raw_doa']).cpu())))

    # update output
    result['model_MAE'] = model_MAE
    result['filter_MAE'] = filter_MAE
    result['gcc_MAE'] = gcc_MAE

    # predicted_classes = np.argmax(belief_over_time, axis=0)
    # predicted_angles = start_angle + angle_step * predicted_classes  # [T]

    return result
    
    

