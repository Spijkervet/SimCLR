import os
import torch
from modules import SimCLR

def load_model(args):
    model = SimCLR(args)
    return model

def save_model(args, model, optimizer):
    out = os.path.join(args.out_dir, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)