conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "data": {
        'dataset_path': "/home/ncusc/gait_data/test/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 74,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (4, 8),
        'restore_iter': 0,
        'total_iter': 160000,
        'margin': 0.2,
        'num_workers': 0,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
