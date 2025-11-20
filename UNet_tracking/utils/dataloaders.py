from torch.utils.data import DataLoader
from utils.speaker_dataset import CostumDataset


def prepare_dataloaders(args, group_name=None):
    dataset_args = dict(
        spec_size=args.spec_size,
        spec_fixed_var=3,
        frame_size=args.frame_size,
        overlap=args.overlap,
        nfft_size=args.nfft,
        ref_channel=3,
        window=args.window,
        rtf_esti_op=args.rtf_esti_method,
        feature_op=args.feature_op,
        num_classes=args.num_classes,
        debug_level=args.debug_level,
        load_rtf=args.load_rtf,
        use_center_crop=args.use_cc,
        transform=None,
        group_name=group_name
    )

    # Create calibration dataset args without group filtering (use all data)
    cal_dataset_args = dataset_args.copy()
    cal_dataset_args['group_name'] = None

    test_dataset = CostumDataset(data_path=args.test_data_path, **dataset_args)
    cal_dataset  = CostumDataset(data_path=args.calib_data_path, **cal_dataset_args)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    cal_loader = DataLoader(cal_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return test_loader, cal_loader
