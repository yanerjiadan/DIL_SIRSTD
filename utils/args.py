import argparse
from datetime import datetime
import os
def get_args():
    parser = argparse.ArgumentParser('Domain Incremental SIRSTD')

    # parser.add_argument('--model_name', type=str, default='DNANet', choices=['DNANet', 'ISNet', 'SCTransNet'])
    parser.add_argument('--dataset', type=str, default='DIL_SIRSTD')
    parser.add_argument('--model_name', type=str, default='MSHNet')
    parser.add_argument('--loss_fn', type=str, default='SoftIoULoss')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--save_dir', type=str, default=None, help='log directory')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--deep_supervision',type=bool, default=True, help='use deep supervision')
    parser.add_argument('--device', type=str, default='cuda', help='number of device')
    parser.add_argument('--mode', type=str, default='train', )
    parser.add_argument('--now_str', type=str, default=None, help='now string')
    
    args = parser.parse_args()
    
    now = datetime.now()
    args.now_str = now.strftime("%Y_%m_%d___%H_%M_%S")
    args.save_dir = f'results/{args.now_str}'
    os.makedirs(args.save_dir, exist_ok=True)
    args_dict = vars(args)
    args_key = list(args_dict.keys())
    args_value = list(args_dict.values())
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        f.write('time:--')
        f.write(args.now_str)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')

    return args