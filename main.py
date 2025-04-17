import os
import numpy as np
import torch
import torchvision
import argparse
import torch.nn.functional as F  # For cosine similarity

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
from simclr.modules.transformations import TransformsSimCLR 
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_model
from utils import yaml_config_hook

def train(args, train_loader, model, criterion, optimizer, writer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        # Positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        
        optimizer.step()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        # Log the loss for every step.
        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch_step", loss.item(), args.global_step)
            args.global_step += 1

        # Log augmented images every 100 steps.
        if args.nr == 0 and step % 100 == 0:
            grid_xi = torchvision.utils.make_grid(x_i[:16], nrow=4, normalize=True)
            grid_xj = torchvision.utils.make_grid(x_j[:16], nrow=4, normalize=True)
            writer.add_image("Augmented Images/x_i", grid_xi, args.global_step)
            writer.add_image("Augmented Images/x_j", grid_xj, args.global_step)

        # Compute and log cosine similarity between positive and negative pair representations.
        with torch.no_grad():
            # Positive cosine similarities.
            cos_sim = F.cosine_similarity(z_i, z_j, dim=-1)
            avg_cos_sim = cos_sim.mean().item()
            writer.add_scalar("Cosine Similarity/avg_positive", avg_cos_sim, args.global_step)
            writer.add_histogram("Cosine Similarity/hist_positive", cos_sim, args.global_step)

            # Compute negative similarities.
            norm_z_i = F.normalize(z_i, dim=1)
            norm_z_j = F.normalize(z_j, dim=1)
            cosine_matrix = torch.mm(norm_z_i, norm_z_j.t())  # (batch_size, batch_size)
            # Mask the diagonal (positive pairs)
            mask = torch.eye(cosine_matrix.size(0), dtype=torch.bool, device=cosine_matrix.device)
            negative_sim = cosine_matrix[~mask].view(cosine_matrix.size(0), -1)
            avg_cos_sim_neg = negative_sim.mean().item()
            writer.add_scalar("Cosine Similarity/avg_negative", avg_cos_sim_neg, args.global_step)
            writer.add_histogram("Cosine Similarity/hist_negative", negative_sim, args.global_step)

            # Log embedding standard deviation as a collapse indicator.
            std_z_i = norm_z_i.std(dim=0).mean().item()
            writer.add_scalar("Embeddings/std_dev", std_z_i, args.global_step)

        loss_epoch += loss.item()
    return loss_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data loading: support for STL10, CIFAR10, and a custom dataset (e.g., dogs vs. cats)
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
    elif args.dataset == "custom":
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.dataset_dir, "dataset/training_set/"),
            transform=TransformsSimCLR(size=args.image_size)
        )
    else:
        raise NotImplementedError

    # Create a DistributedSampler if using more than one node.
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

    # Initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features

    # Initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # Setup optimizer and NT-Xent loss criterion.
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train(args, train_loader, model, criterion, optimizer, writer)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 10 == 0:
            save_model(args, model, optimizer)

        if args.nr == 0:
            avg_loss = loss_epoch / len(train_loader)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {avg_loss}\t lr: {round(lr, 5)}")
            args.current_epoch += 1

            # Log embeddings for a small subset of images.
            model.eval()
            try:
                sample_batch = next(iter(train_loader))
                # Use one view (x_i) for embedding logging.
                sample_images, sample_labels = sample_batch[0][0], sample_batch[1]
                sample_images = sample_images.cuda(non_blocking=True)

                with torch.no_grad():
                    embeddings = model.encoder(sample_images)
                # Convert numerical labels to metadata (e.g., "dog" or "cat")
                #print("Shape of sample images:", sample_images.shape)
                #assert sample_images.ndim == 4, "sample_images must be a 4D tensor [N, C, H, W]"
                cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                metadata = [cifar10_classes[label] for label in sample_labels]
                #metadata = ["dog" if label == 1 else "cat" for label in sample_labels]
                writer.add_embedding(embeddings, metadata=metadata, label_img=sample_images, global_step=epoch, tag="embeddings")
            except Exception as e:
                print(f"Error during embedding logging: {e}")
            model.train()

            # Log gradient histograms for all parameters.
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad, epoch)
        
    # End training: save final model checkpoint.
    save_model(args, model, optimizer)
    
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # Set master address for distributed training (if applicable).
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(f"Training with {args.nodes} nodes, waiting until all nodes join before starting training")
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
