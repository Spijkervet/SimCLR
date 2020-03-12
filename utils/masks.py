import torch

def mask_correlated_samples(args):
    mask = torch.ones((args.batch_size * 2, args.batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(args.batch_size):
        mask[i, args.batch_size + i] = 0
        mask[args.batch_size + i, i] = 0
    return mask
