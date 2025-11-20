import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.metrics import one_hot_to_angle
from utils.gcc_phat import compute_all_gcc_phat, estimate_doa_multiple_micsV2
# from utils.conformal import do_conformal_prediction
from utils.visualize import plot_CP_DOA_results
from utils.utils import write_result_to_txt
from pathlib import Path

def process_sample(model, batch, device, angle_step=5, start_angle=10):
    x_test = batch['input'].to(device=device, dtype=torch.float32)
    target = batch['target'].to(device=device, dtype=torch.float32)
    spp_masks = batch['spp_masks'].to(device=device, dtype=torch.float32)
    raw_input = batch['raw_input'].to(device=device, dtype=torch.float32)
    mic_positions = batch['mic_pos'].to(device=device, dtype=torch.float32)
    raw_doa = batch['raw_doa'].to(device=device, dtype=torch.float32) # use this for trj_endfire_bounce and maybe more

    # Decode ground truth angle 
    target = target[:, :, 0, :]
    gt_angles = one_hot_to_angle(target, angle_step=angle_step)[0]

    # GCC-PHAT estimate
    delays = compute_all_gcc_phat(raw_input, fs=16000, max_tau=None, interp=1)
    gcc_doa = estimate_doa_multiple_micsV2(delays, mic_positions)
    gcc_doa_value = np.mean(list(gcc_doa.values()))

    # Unet model estimate
    pred = model(x_test)
    
    if pred.shape[2] != 1:
        vad_mask = (spp_masks.sum(dim=-1) > 0).cpu().float()
        vad_mask_exp = vad_mask.unsqueeze(1)
        pred = pred.cpu() * vad_mask_exp
        pred = pred.sum(dim=2)
        valid_freqs = vad_mask.sum(dim=1)
        valid_freqs = torch.max(valid_freqs, torch.tensor(1e-6))
        pred = pred / valid_freqs.unsqueeze(1).expand_as(pred)
    else:
        pred = pred.cpu().squeeze(2)

    probabilities = F.softmax(pred, dim=1)
    
    # CP prediction
    # CP_error_esti, CP_class_indices = do_conformal_prediction(probabilities, threshold=threshold)

    angles = np.arange(start_angle, start_angle + angle_step * probabilities.shape[1], angle_step)
    unet_doa_per_frame = [angles[np.argmax(probabilities[0, :, frame])] for frame in range(probabilities.shape[2])]

    model_MAE = np.mean(np.abs(np.stack(unet_doa_per_frame) - gt_angles[1]))
    gcc_MAE = np.mean(np.abs(gcc_doa_value - gt_angles[1]))


    return {
        'probabilities': probabilities,
        'raw_input': raw_input,
        'mic_positions': mic_positions,
        'gt_angles': gt_angles,
        'unet_doa': unet_doa_per_frame,
        'gcc_doa': gcc_doa_value,
        # 'model_MAE': model_MAE,
        # 'gcc_MAE': gcc_MAE,
        # 'CP_error': CP_error_esti,
        # 'CP_sets': CP_class_indices,
        'target': target,
        'raw_doa': raw_doa
    }


def log_and_visualize_sample(
    wandb,
    i,
    result,
    args,
    group_name=None,
    angle_step=5,
    start_angle=10,
    end_angle=170
):
    # Log to Weights & Biases
    wandb.log({ 
        'data path': args.test_data_path,
        'unet error': result['model_MAE'],
        'CP filter error': result['filter_MAE'],
        'CP penalty for jumpy (lambda)': args.lam_sp,
        'CP penalty for size (lambda)': args.lam_si,
        'CP penalty for span (lambda)': args.lam_span,
        # 'gcc-phat error': result['gcc_MAE'],
        # 'CP width': np.mean(result['CP_error']),
        'step': i
    })

    # Optional debug plot
    if args.debug_level == 2:
        selected_frame = 5
        plt.figure(figsize=(10, 6))
        angles = np.arange(start_angle, end_angle, angle_step)
        plt.bar(
            angles,
            result['probabilities'][0, :, selected_frame],
            width=1.5,
            color='skyblue',
            edgecolor='black'
        )
        plt.xlabel("Angles (degrees)")
        plt.title(f"Probabilities of Angles at frame {selected_frame}")
        plt.savefig(f"angle_probabilities_at_frame_{selected_frame}.png")
        plt.close()

    # Print summary
    path = Path(args.test_data_path)
    subfolder = path.parent.name

    # If testing a specific group, create a subfolder for it
    if group_name:
        plot_subfolder = f'{subfolder}/{group_name}'
    else:
        plot_subfolder = subfolder

    save_dir = Path(f'plots/{plot_subfolder}')
    save_dir.mkdir(parents=True, exist_ok=True)

    write_result_to_txt(result, args, filepath=f'plots/{plot_subfolder}/results.txt')

    # =================== plot heat map for all DOAs ===================
    probabilities = result['probabilities']
    target = result['target']
    CP_sets = result['CP_sets']
    # CP_error_esti = result['CP_error']

    CP_map_plt_name = f'plots/{plot_subfolder}/heat_map_with_CP_error_iter_{i}.jpeg'

    # Create the output directory if it doesn't exist
    # save_dir = Path(f'plots/{subfolder}')
    # save_dir.mkdir(parents=True, exist_ok=True)

    belief_over_time = result['belief_over_time']

    plot_CP_DOA_results(probabilities, target, CP_sets, belief_over_time, name = CP_map_plt_name, angle_step=angle_step, debug_level=2)
    wandb.log({'DOA Heatmap': wandb.Image(CP_map_plt_name, caption=f"DOA Probability Heatmap with CP error Iteration {i}")})
    