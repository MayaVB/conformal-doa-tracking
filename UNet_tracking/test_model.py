import argparse
import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys

import wandb
from unet import UNet

from utils.dataloaders import prepare_dataloaders
from utils.visualize import plot_error_vs_true_DOA, plot_histograms_for_all_DOA
from utils.sample_processing import process_sample, log_and_visualize_sample
from utils.conformal import do_conformal_prediction, compute_threshold_from_calibration
from utils.baysian_tracker import build_transition_matrix, track_doa
from utils.utils import export_results_to_csv, write_multi_scenario_results, write_single_scenario_results, get_h5_groups
from evaluate import evaluate_results

def test_model(
        model,
        device,
        test_data_path: str,
        args,
        group_name: str = None
    ):
    """
    Test the trained model on a new dataset.

    :param model: Trained UNet model.
    :param device: Device to run the model on ('cpu' or 'cuda').
    :param test_data_path: Path to the test dataset.
    :param args: Argument namespace containing relevant parameters.
    :param group_name: Specific HDF5 group name to test (optional).
    """

    test_loader, calibration_loader = prepare_dataloaders(args, group_name=group_name)
    
    experiment = wandb.init(project='U-Net-test', resume='allow', anonymous='must')
    experiment.config.update(dict(rtf_esti_method = args.rtf_esti_method, num_classes=args.num_classes, checkpoints_dir= args.checkpoints_dir))   
    
    logging.info(f'''Preprocessing Settings:
        RTF estimation method:  {args.rtf_esti_method}
        Features used:          {args.feature_op}
        Frames size:            {args.frame_size}
        Frames overlap:         {args.overlap}
        Window:                 {args.window}
        NFFT size:              {args.nfft}
        Spectral size:          {args.spec_size}
        Debug level:            {args.debug_level}
        ''')  

    logging.info(f"Starting testing on dataset: {test_data_path}")
    logging.info(f"Model loaded from: {args.checkpoints_dir}")
    
    angle_step = 5
    start_angle = 10
    end_angle = 165
    
    # Before testing loop- for transition languge
    num_classes = 32          # 10°, 15°, ..., 165°
    sigma_deg = 5             # or you can tune this (5, 10, etc.)
    
    doa_mae_list = []
    filter_mae_list = []
    doa_raw_errors = []
    GT_angles_list = []
    model_angles_list = []
    all_coverage_hits = []
    
    model.eval()
    
    with torch.inference_mode():
        
        # Compute conformal prediction threshold using a calibration set
        # threshold = compute_threshold_from_calibration(model, calibration_loader, device="cuda", alpha=0.05)
        # sys.exit()

        threshold = 4.19 #  alpha = 0.1 -> thr = 2.45266 ; alpha = 0.05 -> thr = 3.19444  ;  alpha = 0.02 -> thr = 4.19
        logging.info(f"Computed conformal threshold: {threshold:.4f} deg")

        transition_matrix = build_transition_matrix(num_classes, sigma_deg=sigma_deg, doa_step=angle_step)

        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            
            # process samples
            result = process_sample(model, batch, device, angle_step=angle_step)
            
            # do conformal prediction
            result = do_conformal_prediction(result, threshold=threshold)
            
            # bayesian tracking
            result = track_doa(result, transition_matrix, args.lam_sp, args.lam_si, args.lam_span)

            # eval
            result = evaluate_results(result)
            
            # log and visualize
            log_and_visualize_sample(wandb, i, result, args, group_name=group_name, angle_step=angle_step, start_angle=start_angle, end_angle=end_angle)
            
            GT_angles_list.append(result['gt_angles'][0])
            doa_mae_list.append(result['model_MAE'])
            filter_mae_list.append(result['filter_MAE'])
            model_angles_list.append(result['unet_doa'])
            all_coverage_hits.extend(result.get('coverage_hits', []))

            # Collect raw errors for proper MSE calculation
            unet_doa = np.stack(result['unet_doa'])
            gt_angles = result['raw_doa'].cpu().numpy()
            raw_errors = unet_doa - gt_angles
            doa_raw_errors.extend(raw_errors)
    
    # =================== plot error vs true DOA ===================
    path = Path(args.test_data_path)
    subfolder = path.parent.name
    # If testing a specific group, create a subfolder for it
    if group_name:
        plot_subfolder = f'{subfolder}/{group_name}'
    else:
        plot_subfolder = subfolder

    error_vs_true_doa_plt_name = f'plots/{plot_subfolder}/DOA_error_vs_true_DOA.jpeg'
    # error_vs_true_doa_plt_name = 'plots/test_latest/DOA error vs true DOA.jpeg'
    # plot_error_vs_true_DOA(GT_angles_list, doa_mae_list, name=error_vs_true_doa_plt_name)
    # wandb.log({'DOA eror vs GT': wandb.Image(error_vs_true_doa_plt_name, caption=f"DOA error vs true DOA")})

    # =================== hisograms ===================
    Path(f'plots/{plot_subfolder}').mkdir(parents=True, exist_ok=True)
    hist_vs_true_doa_plt_name = f'plots/{plot_subfolder}/histograms_vs_true_doa.jpeg'
    plot_histograms_for_all_DOA(GT_angles_list, model_angles_list, name=hist_vs_true_doa_plt_name)

    # Calculate metrics
    unet_mae = np.mean(doa_mae_list)
    filter_mae = np.mean(filter_mae_list)
    
    # Flatten all raw errors into a single array for MSE calculation
    flattened_raw_errors = np.concatenate([np.array(err).flatten() for err in doa_raw_errors])
    doa_mse = np.mean(flattened_raw_errors ** 2) 
    cp_coverage = np.mean(all_coverage_hits) if all_coverage_hits else 0.0

    logging.info(f"Testing completed! UNet MAE: {unet_mae:.3f}, Filter MAE: {filter_mae:.3f}, MSE: {doa_mse:.3f}, CP Coverage: {cp_coverage:.3f}")

    scenario_name = group_name if group_name else Path(test_data_path).parent.name

    return {
        'scenario': scenario_name,
        'unet_mae': unet_mae,
        'filter_mae': filter_mae,
        'doa_mse': doa_mse,
        'cp_coverage': cp_coverage,
        'num_samples': len(doa_mae_list),
        'rtf_esti_method': args.rtf_esti_method,
        'feature_op': args.feature_op,
        'num_classes': args.num_classes,
        'lambda_spacing': args.lam_sp,
        'lambda_size': args.lam_si,
        'lambda_span': args.lam_span,
        'threshold': threshold,
        'checkpoints_dir': args.checkpoints_dir
    }


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--mode', '-MODE', choices=['train', 'test'], default='test', help="train or test", dest='mode')
    parser.add_argument('--save-checkpoint', metavar='SAV', type=bool, default=False, help='no saving model in test', dest='save')
    parser.add_argument('--load', '-f', type=str, default=True, help='always load model in test mode (.pth file)')
    parser.add_argument('--classes', '-c', type=int, default=32, help='Number of classes', dest='num_classes') # (165-10)/5 + 1
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    parser.add_argument('--features', '-feat', choices=['ReImWithoutSpec', 'ReImWithSpec', 'SinCosWithoutSpec', 'SinCosWithSpec'], 
                        default='ReImWithoutSpec', help='features used', dest='feature_op')
    parser.add_argument('--rtf-estimation-method', '-rem', choices=['EVD', 'iRTF'], default='iRTF', help="select RTF estimation method", dest='rtf_esti_method')
    
    parser.add_argument('--spectral-size', '-ss', type=float, default=128, help='spectral size selection', dest='spec_size')
    parser.add_argument('--frame-size', '-fs', type=float, default=512, help='frame size', dest='frame_size')
    parser.add_argument('--overlap', '-o', type=float, default=0.75, help='frame overlap valuse in presentage', dest='overlap')
    parser.add_argument('--nfft-size', '-nfft', type=float, default=1024, help='frame overlap valuse in presentage', dest='nfft')
    parser.add_argument('--window', '-w', metavar='WIN', choices=['hann', 'hamming'], default='hann', help='window used on frames', dest='window')
    parser.add_argument('--load-rtf', '-lrtf', metavar='LRTF', type=bool, default=False, help='load rft from saved', dest='load_rtf')
    parser.add_argument('--use-center-crop', '-cc', metavar='UCC', type=bool, default=False, help='use center crop- for equal batch size', dest='use_cc')

    parser.add_argument('--lambda-spacing', '-LamSp', metavar='Lspace', type=float, default=0.5, help='penalty for CP jumpyness', dest='lam_sp')
    parser.add_argument('--lambda-size', '-LamSi', metavar='Lsize', type=float, default=0.5, help='penalty for CP size', dest='lam_si')
    parser.add_argument('--lambda-span', '-LamSpan', metavar='Lspan', type=float, default=1, help='penalty for CP span', dest='lam_span')

    parser.add_argument('--checkpoints-dir', '-checkdir', metavar='CheckDIR', type=str, default= 'checkpoints_baseline', help='path for model loading. checkpoints_sav_fc', dest='checkpoints_dir')
    parser.add_argument('--test-data-path', '-datapath', metavar='DataPath', type=str, default= '../dataset_folder/gannot-lab/gannot-lab1/SpeakerLocGen/test4paper/train_dynamic_0.2rt_50scenes/RevMovingSrcDataset.h5', help='path for test data', dest='test_data_path')
    # parser.add_argument('--test-data-path', '-datapath', metavar='DataPath', type=str, default= '../dataset_folder/gannot-lab/gannot-lab1/SpeakerLocGen/test_static_rt0.2/RevMovingSrcDataset.h5', help='path for test data', dest='test_data_path')
    parser.add_argument('--calib-data-path', '-calibpath', metavar='CalibPath', type=str, default= '../dataset_folder/gannot-lab/gannot-lab1/SpeakerLocGen/calib/RevMovingSrcDataset.h5', help='path for calibration data', dest='calib_data_path')
    # parser.add_argument('--calib-data-path', '-calibpath', metavar='CalibPath', type=str, default= '../dataset_folder/gannot-lab/gannot-lab1/SpeakerLocGen/calib_new/static_w_burst_rt05/RevMovingSrcDataset.h5', help='path for calibration data', dest='calib_data_path')
    parser.add_argument('--debug-level', '-debug', type=int, default=0, help='[0-low debug, 1-mid debug, 2-high debug] detemine the amout of plots and prints the script outputs', dest='debug_level')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_of_mics = 5
    num_channels = 2*(num_of_mics - 1) # num of features (3 for RGB images)
    
    model = UNet(n_channels=num_channels, n_classes=args.num_classes, bilinear=args.bilinear)
    model.to(device=device)
    
    wandb.login()

    selected_checkpoint = None
    if args.load:
        files = [f for f in os.listdir(args.checkpoints_dir) if f.endswith(('.pth', '.ckpt'))]
        selected_checkpoint = max((os.path.join(args.checkpoints_dir, f) for f in files), key=os.path.getmtime, default=None)

        logging.info(f"Model loaded from: {selected_checkpoint}")
        model.load_state_dict(torch.load(selected_checkpoint, map_location=device, weights_only=False))

    if args.mode == 'test':
        # Check if there are multiple groups in the HDF5 file
        groups = get_h5_groups(args.test_data_path)

        if len(groups) > 1:
            # Test all groups
            results = []
            logging.info(f"Found {len(groups)} groups. Testing all scenarios...")

            for group_name in groups:
                logging.info(f"\n{'='*50}")
                logging.info(f"Testing group: {group_name}")
                logging.info(f"{'='*50}")

                result = test_model(model, device, args.test_data_path, args, group_name)
                results.append(result)

                logging.info(f"Group '{group_name}' completed - UNet MAE: {result['unet_mae']:.3f}, Filter MAE: {result['filter_mae']:.3f}, MSE: {result['doa_mse']:.3f}, CP Coverage: {result['cp_coverage']:.3f}")

            # Export all results to CSV
            csv_file = export_results_to_csv(results, args.test_data_path)

            # Write logging and results.txt using the new function
            write_multi_scenario_results(results, args, csv_file, selected_checkpoint)

        else:
            # Test single scenario (one group or no groups)
            result = test_model(model, device, args.test_data_path, args)

            # Export single result to CSV
            csv_file = export_results_to_csv([result], args.test_data_path)

            # Write logging and results.txt using the new function
            write_single_scenario_results(result, args, selected_checkpoint)

    else:  # Train the model
        print('maybe merge train and test scripts!!')