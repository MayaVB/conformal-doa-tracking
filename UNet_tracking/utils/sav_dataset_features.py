import numpy as np
from utils.speaker_dataset import CostumDataset
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')

    parser.add_argument('--features', '-feat', choices=['ReImWithoutSpec', 'ReImWithSpec', 'SinCosWithoutSpec', 'SinCosWithSpec'], 
                        default='ReImWithoutSpec', help='features used', dest='feature_op')

    parser.add_argument('--rtf-estimation-method', '-rem', choices=['EVD', 'iRTF'], default='iRTF', help="select RTF estimation method", dest='rtf_esti_method')
    parser.add_argument('--doa-grid-res', '-res', type=int, default=2, help='doa grid search in deg', dest='res')
    parser.add_argument('--spectral-size', '-ss', type=float, default=128, help='spectral size selection', dest='spec_size')
    parser.add_argument('--frame-size', '-fs', type=float, default=512, help='frame size', dest='frame_size')
    parser.add_argument('--overlap', '-o', type=float, default=0.75, help='frame overlap value in percentage', dest='overlap')
    parser.add_argument('--nfft-size', '-nfft', type=float, default=1024, help='frame overlap value in percentage', dest='nfft')
    parser.add_argument('--window', '-w', metavar='WIN', choices=['hann', 'hamming'], default='hann', help='window used on frames', dest='window')
    parser.add_argument('--use-context-frame', '-ucm', metavar='ucm', type=bool, default=False, help='True/False use context frames', dest='use_cm')
    parser.add_argument('--num-context-frame', '-ncm', metavar='ncm', type=int, default=10, help='number od context frames', dest='num_cm')
    parser.add_argument('--load-rtf', '-lrtf', metavar='LRTF', type=bool, default=False, help='load rft from saved', dest='load_rtf')
    parser.add_argument('--debug-level', '-debug', type=int, default=0, help='[0-low debug, 1-mid debug, 2-high debug] determine the amount of plots and prints the script outputs', dest='debug_level')
    parser.add_argument('--classes', '-c', type=int, default=32, help='Number of classes [32, 78, 90]', dest='num_classes') # (165-10)/5 + 1


    return parser.parse_args()


def save_args_to_txt(args, target_file):
    # Extract the base name of the target file without extension
    base_name = os.path.splitext(target_file)[0]
    txt_file_path = f"{base_name}_args.txt"

    # Save arguments to the .txt file
    with open(txt_file_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Arguments saved to {txt_file_path}")


if __name__ == '__main__':
    args = get_args()


    # data_path = '../dataset_folder/train/RevMovingSrcDataset.h5'
    data_path = '/home/dsi/mayavb/PythonProjects/SpeakerLocGen/biger_train/test/RevMovingSrcDataset.h5'
    # target_file = 'data/rtf_features_database/rtf_data_cm_true.h5'  # Add all your target file paths here
    target_file = 'data/rtf_features_database/rtf_baseline.h5'  # Add all your target file paths here

    # # Instantiate the dataset
    # dataset = CostumDataset(data_path=data_path, 
    #                         spec_size=128, 
    #                         spec_fixed_var=3, 
    #                         target_file=target_file)
        
    dataset = CostumDataset(data_path=data_path, 
                            spec_size = args.spec_size,
                            spec_fixed_var = 3,
                            frame_size = args.frame_size,
                            overlap = args.overlap,
                            nfft_size = args.nfft,
                            ref_channel = 3,
                            window = args.window,
                            rtf_esti_op = args.rtf_esti_method,
                            use_context_frame = args.use_cm,
                            num_context_frame = args.num_cm,
                            feature_op = args.feature_op,
                            num_classes=args.num_classes,
                            transform = None,
                            load_rtf = args.load_rtf,
                            target_file = target_file,
                            debug_level = args.debug_level
                            )
    
    # Save all processed features
    dataset.process_and_save_features()
    
    # Save the arguments to a .txt file
    save_args_to_txt(args, target_file)
    
    
# import numpy as np
# from utils.speaker_dataset import CostumDataset


# if __name__ == "__main__":
#     data_path = '../dataset_folder/train/RevMovingSrcDataset.h5'
#     #data_path = '/home/dsi/mayavb/PythonProjects/SpeakerLocGen/train/RevMovingSrcDataset.h5'

#     target_file = 'data/rtf_data.h5'  # Add all your target file paths here

#     # Instantiate the dataset
#     dataset = CostumDataset(data_path=data_path, spec_size=128, spec_fixed_var=3, target_file=target_file)
        
#     # Save all processed features
#     dataset.process_and_save_features()