import argparse

def get_args():
    parser = argparse.ArgumentParser('Domain Incremental SIRSTD')

    parser.add_argument('--model_name', type=str, default='DNANet', choices=['DNANet', 'ISNet', 'SCTransNet'])
    parser.add_argument('--dataset', type=str, default='DIL-SIRSTD')
    parser.add_argument('--model_name', type=str, default='MSHNet')
    parser.add_argument('--loss_fn', type=str, default='SoftIoULoss')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer')
    parser.add_argument('--deep_supervision',type=bool, default=True, help='use deep supervision')
    parser.add_argument('--device', type=str, default='cuda:0', help='number of device')

    args = parser.parse_args()
    return args