import os
import numpy as np
import torch
import torchvision
import argparse
import time

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from model import load_model, save_model
from modules import NT_Xent
from modules.sync_batchnorm import convert_model
from modules.transformations import TransformsSimCLR
from utils import yaml_config_hook


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += loss.item()
        args.global_step += 1
    return loss_epoch

def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(gpu)

    root = "./datasets"
    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root, split="unlabeled", download=True, transform=TransformsSimCLR(size=args.image_size)
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimCLR(size=args.image_size)
        )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    model, optimizer, scheduler = load_model(args, train_loader)

    if args.nodes > 1:
        model = DDP(model, device_ids=[gpu])

    criterion = NT_Xent(args.batch_size, args.temperature, args.device)

    writer = SummaryWriter()
    args.out_dir = writer.log_dir
    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        t0 = time.time()
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)
        print(time.time() - t0)

        if scheduler:
            scheduler.step()

        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        writer.add_scalar("Misc/learning_rate", lr, epoch)
        print(
            f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
        )
        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "5000"


    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(f"Training with {args.nodes} nodes, waiting until all nodes join before starting training")
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)