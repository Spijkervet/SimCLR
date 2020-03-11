import torch
import torchvision
import argparse

from experiment import ex
from model import load_model
from utils import post_config_hook


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)
            # h = 512
            # z = 64

        output = model(h)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

def test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        # get encoding
        with torch.no_grad():
            h, z = simclr_model(x)
            # h = 512
            # z = 64

        output = model(h)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch

@ex.automain
def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args = post_config_hook(args, _run)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "./datasets"
    simclr_model = load_model(args, reload_model=True)
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()


    ## Logistic Regression
    model = torch.nn.Sequential(torch.nn.Linear(simclr_model.n_features, 10)).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = torchvision.datasets.STL10(
        root, split="train", download=True, transform=torchvision.transforms.ToTensor()
    )

    test_dataset = torchvision.datasets.STL10(
        root, split="test", download=True, transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        drop_last=True,
        num_workers=args.workers,
    )

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = train(args, train_loader, simclr_model, model, criterion, optimizer)
        print(f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

    # final testing
    loss_epoch, accuracy_epoch = test(args, test_loader, simclr_model, model, criterion, optimizer)
    print(f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}")
