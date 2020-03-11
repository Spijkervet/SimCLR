import os
import torch
import torchvision
import argparse

from torch.utils.tensorboard import SummaryWriter


from model import load_model, save_model

#### pass configuration
from experiment import ex


class TransformsSinCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size=96),
                torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


def nt_xent(args, z_i, z_j, cossim, mask, criterion):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """

    p1 = torch.cat((z_i, z_j), dim=0)
    sim = cossim(p1.unsqueeze(1), p1.unsqueeze(0)) / args.temperature

    sim_i_j = torch.diag(sim, args.batch_size)
    sim_j_i = torch.diag(sim, -args.batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(args.batch_size * 2, 1)
    negative_samples = sim[mask].reshape(args.batch_size * 2, -1)

    labels = torch.zeros(args.batch_size * 2).to(args.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = criterion(logits, labels)
    loss /= 2 * args.batch_size
    return loss


@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)

    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    else:
        out_dir = _run.observers[0].dir

    args.out_dir = out_dir

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "./datasets"
    model = load_model(args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS

    train_sampler = None
    train_dataset = torchvision.datasets.STL10(
        root, split="unlabeled", download=True, transform=TransformsSinCLR()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    tb_dir = os.path.join(args.out_dir, _run.experiment_info["name"])
    os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)

    cossim = torch.nn.CosineSimilarity(dim=2)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    mask = torch.ones((args.batch_size * 2, args.batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(args.batch_size):
        mask[i, args.batch_size + i] = 0
        mask[args.batch_size + i, i] = 0

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = 0
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_i = x_i.to(args.device)
            x_j = x_j.to(args.device)

            # positive pair, with encoding
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)

            loss = nt_xent(args, z_i, z_j, cossim, mask, criterion)

            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            loss_epoch += loss.item()
            args.global_step += 1

        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")
        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)
