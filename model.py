import os
import torch
import pandas as pd

from simclr import SimCLR
from simclr.modules import LARS


def load_optimizer(args, model, add_params_model=None):

    scheduler = None
    if add_params_model is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(add_params_model.parameters())

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=3e-4)  # TODO: LARS
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        learning_rate = 0.3 * args.batch_size / 256
        optimizer = LARS(
            params,
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

    return optimizer, scheduler


def save_model(args, model, optimizer):
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

def save_classif_model(args, model):
    out = os.path.join(args.model_path, "classif/classif_checkpoint_{}.tar".format(args.current_epoch))
    if not os.path.exists(os.path.join(args.model_path,"classif/")):
        os.mkdir(os.path.join(args.model_path,"classif/"))
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

def weights_onnpu(args, weight):
    out = os.path.join(args.model_path, "weights_onnpu/weights.pkl")

    weight = float(weight.detach().cpu().numpy())
    if not os.path.exists(os.path.join(args.model_path, "weights_onnpu/")):
        os.mkdir(os.path.join(args.model_path,"weights_onnpu/"))
    if args.current_epoch==0:
        sweights = pd.Series({args.current_epoch: weight})
        sweights.to_pickle(out)
    else:
        sweights = pd.read_pickle(out)
        sweights = sweights.append(pd.Series({args.current_epoch: weight}))
        sweights.to_pickle(out)


