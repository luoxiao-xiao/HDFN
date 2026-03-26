import argparse
import multiprocessing
from run import HDFN_run


def parse_args():
    parser = argparse.ArgumentParser(description='HDFN: Hierarchical Dynamic Fusion Network')
    parser.add_argument('--mode',    type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test (default: train)')
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'],
                        help='Dataset name (default: mosi)')
    parser.add_argument('--seeds',   type=int, nargs='+', default=[233],
                        help='Random seeds (default: 233)')
    parser.add_argument('--model_save_dir', type=str, default='./pt',
                        help='Directory to save model checkpoints (default: ./pt)')
    parser.add_argument('--res_save_dir',   type=str, default='./result',
                        help='Directory to save results (default: ./result)')
    parser.add_argument('--log_dir',        type=str, default='./log',
                        help='Directory to save logs (default: ./log)')
    return parser.parse_args()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()

    is_training = (args.mode == 'train')

    print(f"[HDFN] Mode: {args.mode.upper()} | Dataset: {args.dataset} | Seeds: {args.seeds}")

    HDFN_run(
        model_name='HDFN',
        dataset_name=args.dataset,
        is_tune=False,
        seeds=args.seeds,
        model_save_dir=args.model_save_dir,
        res_save_dir=args.res_save_dir,
        log_dir=args.log_dir,
        mode=args.mode,
        is_training=is_training
    )
