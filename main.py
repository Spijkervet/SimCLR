import os
import torch
import torchvision
import argparse

from torch.utils.tensorboard import SummaryWriter


from model import load_model, save_model

#### pass configuration
from experiment import ex


class TransformsSinCLR():
    """
    A stochastic data augmentation module that transforms any given data example randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self):
        s = 1
        resize_crop = torchvision.transforms.RandomResizedCrop(size=96)  # crop_size
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        rnd_color_jitter = torchvision.transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = torchvision.transforms.RandomGrayscale(p=0.2)
        self.train_transform = torchvision.transforms.Compose(
            [
                resize_crop,
                torchvision.transforms.RandomHorizontalFlip(), # with 0.5 probability
                rnd_color_jitter,
                rnd_gray,
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


def calc_loss(args, z_i, z_j, cossim, mask_diag, mask_triag):

    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """

    ## pairwise similarity, manually:
    p1 = torch.cat((z_i, z_j), dim=0)
    # sim = torch.zeros((all_pairs.size(0), all_pairs.size(0))).to(args.device)
    # for i_idx, i in enumerate(all_pairs):
    #     for j_idx, j in enumerate(all_pairs):
    #         i = i.unsqueeze(0)
    #         j = j.unsqueeze(1)
    #         mm = torch.matmul(i, j).squeeze().to(args.device)
    #         sim[i_idx, j_idx] = (mm / args.temperature).exp()
    # sim = torch.matmul(p1, p1.T)

    sim = cossim(p1.unsqueeze(1), p1.unsqueeze(0))
    sim /= args.temperature
    sim = torch.exp(sim)

    nominator = torch.cat(
        (
            sim[torch.arange(0, args.batch_size), mask_triag],
            sim[mask_triag, torch.arange(0, args.batch_size)],
        ),
        dim=0,
    )

    masks = sim.clone()
    masks[mask_diag, mask_diag] = 0.0
    masks[mask_triag, torch.arange(0, args.batch_size)] = 0.0
    masks[torch.arange(0, args.batch_size), mask_triag] = 0.0
    denominator = masks.sum()

    l_ij = -torch.log(nominator / denominator).to(args.device)
    total_loss = l_ij.mean()
    total_loss /= 2
    return total_loss

@ex.automain
def main(_run, _log):
    torch.autograd.set_detect_anomaly(True)
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

    cos_2 = torch.nn.CosineSimilarity(dim=2)
    mask_diag = torch.arange(0, args.batch_size * 2).to(args.device)
    mask_triag = torch.arange(args.batch_size, args.batch_size * 2).to(args.device)
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

            loss = calc_loss(args, z_i, z_j, cos_2, mask_diag, mask_triag)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")
            
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            loss_epoch += loss.item()
            args.global_step += 1
        
        writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")
        if epoch % 10 == 0:
            save_model(args, model, optimizer)

        args.current_epoch += 1 