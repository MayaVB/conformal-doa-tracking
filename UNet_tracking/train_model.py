import argparse
import logging

import torch
import wandb

import numpy as np
import torch.nn.functional as F

from torch import optim
from tqdm import tqdm
from unet import UNet
from pathlib import Path

from evaluate import eval_net
from utils.metrics import accuracy_at_k, mean_absolute_error
from utils.speaker_dataset import CostumDataset
from utils.utils import set_seed
from torch.utils.data import DataLoader, random_split


# dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        device,
        args,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        center_crop: int = 5,
        val_percent: float = 0.1,
        save_checkpoint: bool = False,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        LROnPlateau_patience: int = 5,
        LROnPlateau_factor = 0.5,
        momentum: float = 0.999,
        gradient_clipping: float = 0.5,
        profile: bool = True,  # New argument to control profiling
        profiling_dir: str = './profiler_results'  # Directory to save profiling results
        ):
    
# data_path = 
# '../dataset_folder/train/RevMovingSrcDataset.h5'
# '/home/dsi/mayavb/PythonProjects/SpeakerLocGen/train/RevMovingSrcDataset.h5'
# '/home/dsi/mayavb/PythonProjects/SpeakerLocGen/train_base/RevMovingSrcDataset.h5'
# '../dataset_folder/train_base/RevMovingSrcDataset.h5'
# '/home/dsi/mayavb/PythonProjects/SpeakerLocGen/train/RevMovingSrcDatasetWavs'
# '../dataset_folder/train/RevMovingSrcDatasetWavs'
# /home/dsi/mayavb/PythonProjects/SpeakerLocGen/biger_train/test/RevMovingSrcDataset.h5'

    # 1. Create dataset           
    dataset = CostumDataset(data_path = '../dataset_folder/train/RevMovingSrcDataset.h5', #data_path = '../dataset_folder/train/RevMovingSrcDataset.h5', data_path = '/home/dsi/mayavb/PythonProjects/SpeakerLocGen/train/RevMovingSrcDataset.h5'
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
                            target_file = args.target_file,
                            debug_level = args.debug_level
                            )

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    loader_args = dict(batch_size=batch_size, num_workers=30, pin_memory=True) # num_workers=48
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')

    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            val_percent=val_percent, rtf_esti_method = args.rtf_esti_method, 
            load_features=args.load_rtf, load_features_path = args.target_file, save_checkpoint=save_checkpoint,
            save_checkpoint_path=args.dir_checkpoint)
        )   
    
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

    logging.info(f'''Starting training:
        Epochs:             {epochs}
        Batch size:         {batch_size}
        Learning rate:      {learning_rate}
        Patience:           {LROnPlateau_patience}
        LROnPlateau factor: {LROnPlateau_factor}
        Gradiant clipping:  {gradient_clipping}
        Weight Decay:       {weight_decay}
        Number of Classes:  {args.num_classes}
        Training size:      {n_train}
        Validation size:    {n_val}
        Checkpoints:        {save_checkpoint}
        Device:             {device.type}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Add weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Add weight decay for regularization

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=LROnPlateau_patience, factor=LROnPlateau_factor, min_lr=1e-5)
    
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()

        loss_list = []
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                x_train = batch['input']
                target = batch['target']
                
                use_spp = False
                spp_masks =  batch['spp_masks']

                assert x_train.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {x_train.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
            
                x_train = x_train.to(device=device, dtype=torch.float32)

                # Adjust target type based on number of classes
                target = target.to(device=device, dtype=torch.float32)
                pred = model(x_train)
                
                if pred.shape[2] == 1: # if not use_spp:
                    target = torch.mean(target, dim=2)  # Average over the freqs- they are all zeros anyway

                # loss calculation for multi-class classification
                if pred.shape[2] != 1: # use_spp:
                    loss = criterion(pred, target)
                else:
                    loss = criterion(pred.squeeze(2), target)

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(x_train.shape[0])
                global_step += 1

                loss_list.append(loss.item())
                
                experiment.log({ # this is shit graph this is the last loss
                    'iteration loss': loss.item(),
                    'step': global_step
                    #'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Evaluation round
                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            data = value.data.cpu()
                            data_range = data.max() - data.min()
                            num_bins = min(50, max(10, int(data_range / 1e-3)))  # Ensure bins are meaningful
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu(), num_bins=num_bins)

                        val_score = eval_net(model, val_loader, device, n_val)
                        
                        with torch.inference_mode():
                            predictions = pred.cpu().squeeze(2)
                            probabilities = F.softmax(predictions, dim=1)
                        
                            topKvalue = 5
                            acc_top_k = accuracy_at_k(probabilities, target, k=topKvalue)
                            logging.info(f'Accuracy Top k={topKvalue}: {acc_top_k:.4f}')
                            experiment.log({f'Accuracy top k={topKvalue}:': acc_top_k, 'step': global_step})

                            MAE_result = mean_absolute_error(probabilities, target, angle_step=args.res)
                            logging.info(f'MAE: {MAE_result:.4f}')
                            experiment.log({f'MAE:': MAE_result, 'step': global_step})
                                                    
                            logging.info('Validation score: {}'.format(val_score))
                            experiment.log({'Validation score': val_score, 'step': global_step})
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation score': val_score,
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass
                        
        experiment.log({'train loss': np.mean(loss_list), 'epoch': epoch})
        
        if save_checkpoint and (epoch % 20 == 0):            
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(Path(args.dir_checkpoint) / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=140, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size 40')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', metavar='WD', type=float, default=1e-4, help='optim.Adam- Weight decay', dest='wd')
    parser.add_argument('--patience', metavar='P', type=int, default=5, help='ReduceLROnPlateau- patience', dest='patience')
    parser.add_argument('--factor', metavar='F', type=float, default=0.1, help='ReduceLROnPlateau- factor', dest='factor')
    parser.add_argument('--save-checkpoint', metavar='SAV', type=bool, default=True, help='saving model in checkpoints', dest='save')
    parser.add_argument('--save-checkpoint-dir', metavar='SAV-path', type=str, default='./checkpoints_baseline_with_seed/', help='checkpoints saving path', dest='dir_checkpoint')
    parser.add_argument('--load-rtf', '-lrtf', metavar='LRTF', type=bool, default=False, help='load rft from saved', dest='load_rtf')
    parser.add_argument('--load-rtf-path', '-lrtfp', metavar='LRTFP', type=str, default='data/rtf_features_database/rtf_data_EVD.h5', help='load rft data path', dest='target_file')

    parser.add_argument('--gradient-clipping', '-g', dest='gradient_clipping', metavar='B', type=float, default=0.8)
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=32, help='Number of classes', dest='num_classes') # (165-10)/5 + 1
    parser.add_argument('--features', '-feat', choices=['ReImWithoutSpec', 'ReImWithSpec', 'SinCosWithoutSpec', 'SinCosWithSpec'], 
                        default='ReImWithoutSpec', help='features used', dest='feature_op')

    parser.add_argument('--rtf-estimation-method', '-rem', choices=['EVD', 'iRTF'], default='iRTF', help="select RTF estimation method", dest='rtf_esti_method')
    parser.add_argument('--doa-grid-res', '-res', type=int, default=5, help='doa grid search in deg', dest='res')
    parser.add_argument('--spectral-size', '-ss', type=float, default=128, help='spectral size selection', dest='spec_size')
    parser.add_argument('--frame-size', '-fs', type=float, default=512, help='frame size', dest='frame_size')
    parser.add_argument('--overlap', '-o', type=float, default=0.75, help='frame overlap valuse in presentage', dest='overlap')
    parser.add_argument('--nfft-size', '-nfft', type=float, default=1024, help='frame overlap valuse in presentage', dest='nfft')
    parser.add_argument('--window', '-w', metavar='WIN', choices=['hann', 'hamming'], default='hann', help='window used on frames', dest='window')
    parser.add_argument('--use-context-frame', '-ucm', metavar='ucm', type=bool, default=False, help='True/False use context frames', dest='use_cm')
    parser.add_argument('--num-context-frame', '-ncm', metavar='ncm', type=int, default=10, help='number od context frames', dest='num_cm')

    parser.add_argument('--debug-level', '-debug', type=int, default=0, help='[0-low debug, 1-mid debug, 2-high debug] detemine the amout of plots and prints the script outputs', dest='debug_level')


    return parser.parse_args()


if __name__ == '__main__':
    
    set_seed(42)

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    num_of_mics = 5
    num_classes = args.num_classes # doa classification vector (number of probabilities you want to get per pixel)
    num_channels = 2*(num_of_mics - 1) # num of features (3 for RGB images)
    
    model = UNet(n_channels=num_channels, n_classes=num_classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    
    try:
        wandb.login()
        train_model(
            model=model,
            args=args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            save_checkpoint=args.save,
            weight_decay= args.wd,
            LROnPlateau_patience= args.patience,
            LROnPlateau_factor= args.factor,
            gradient_clipping = args.gradient_clipping
        )
        
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                    'Enabling checkpointing to reduce memory usage, but this slows down training. '
                    'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            args=args,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            center_crop=args.cc,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

