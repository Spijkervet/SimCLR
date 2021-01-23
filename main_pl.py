import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule


# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from utils import yaml_config_hook


class ContrastiveLearning(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.hparams = args

        # initialize ResNet
        self.encoder = get_resnet(self.hparams.resnet, pretrained=False)
        self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(self.encoder, self.hparams.projection_dim, self.n_features)
        self.criterion = NT_Xent(
            self.hparams.batch_size, self.hparams.temperature, world_size=1
        )

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        (x_i, x_j), _ = batch
        loss = self.forward(x_i, x_j)
        return loss

    def configure_criterion(self):
        criterion = NT_Xent(self.hparams.batch_size, self.hparams.temperature)
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        elif self.hparams.optimizer == "LARS":
            # optimized using LARS with linear learning rate scaling
            # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
            learning_rate = 0.3 * args.batch_size / 256
            optimizer = LARS(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=args.weight_decay,
                exclude_from_weight_decay=["batch_normalization", "bias"],
            )

            # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, args.epochs, eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        raise NotImplementedError
    
    if args.gpus == 1:
        workers = args.workers
    else:
        workers = 0

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=workers)

    cl = ContrastiveLearning(args)
    
    trainer = Trainer.from_argparse_args(args)
    trainer.sync_batchnorm=True
    trainer.fit(cl, train_loader)
