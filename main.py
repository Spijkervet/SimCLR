import os
import torch
import torchvision
import argparse

from torch.utils.tensorboard import SummaryWriter

apex = False
try:
    from apex import amp
    apex = True
except ImportError:
    print(
        "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
    )

from model import load_model, save_model
from modules import NT_Xent
from modules.transformations import TransformsSimCLR
from utils import post_config_hook

#### pass configuration
from experiment import ex


def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = criterion(z_i, z_j)

        if apex and args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += loss.item()
        args.global_step += 1
    return loss_epoch


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "./datasets"

    train_sampler = None

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            root, split="unlabeled", download=True, transform=TransformsSimCLR(size=96)
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=TransformsSimCLR(size=32)
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    model, optimizer, scheduler = load_model(args, train_loader)

    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    criterion = NT_Xent(args.batch_size, args.temperature, args.device)

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

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
