import os
import torch
from modules import SimCLR

def load_model(args, reload_model=False):
    model = SimCLR(args)

    if reload_model:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.model_num))
        model.load_state_dict(torch.load(model_fp))
    
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # TODO: LARS
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
            )

        print("### USING FP16 ###")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    return model, optimizer


def save_model(args, model, optimizer):
    out = os.path.join(args.out_dir, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)