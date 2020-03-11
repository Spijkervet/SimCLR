import os
import torch
from modules import SimCLR

def load_model(args, reload_model=False):
    model = SimCLR(args)

    if reload_model:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.model_num))
        model.load_state_dict(torch.load(model_fp))
    return model


def save_model(args, model, optimizer):
    out = os.path.join(args.out_dir, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)