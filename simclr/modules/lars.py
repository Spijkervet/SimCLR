from torch.optim.optimizer import Optimizer, required
import re
import torch

EETA_DEFAULT = 0.001

class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        param_names,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        param_names: names of parameters of model obtained by
            [name for name, p in model.named_parameters() if p.requires_grad]
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['bn', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            param_names = param_names,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.param_names = param_names
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

        self.param_name_map = {'batch_normalization':'bn','bias':'bias'}

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]
            #param_names = group["param_names"]

            for p_name, p in zip(group["param_names"],group["params"]):
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                if self._use_weight_decay(p_name):
                    grad += self.weight_decay * param
                else:
                    print(p_name)

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    if self._do_layer_adaptation(p_name):
                        w_norm = torch.norm(param)
                        g_norm = torch.norm(grad)

                        device = g_norm.get_device()
                        trust_ratio = torch.where(
                            w_norm.gt(0),
                            torch.where(
                                g_norm.gt(0),
                                (self.eeta * w_norm / g_norm),
                                torch.Tensor([1.0]).to(device),
                            ),
                            torch.Tensor([1.0]).to(device),
                        ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(grad, alpha = scaled_lr)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    trust_ratio = 1.0

                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + grad
                    else:
                        update = next_v

                    if self._do_layer_adaptation(param_name):
                        w_norm = torch.norm(param)
                        g_norm = torch.norm(grad)

                        device = g_norm.get_device()
                        trust_ratio = torch.where(
                            w_norm.gt(0),
                            torch.where(
                                g_norm.gt(0),
                                (self.eeta * w_norm / g_norm),
                                torch.Tensor([1.0]).to(device),
                            ),
                            torch.Tensor([1.0]).to(device),
                        ).item()

                    scaled_lr = lr * trust_ratio

                    p.data.add_(-update, alpha = scaled_lr)

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(self.param_name_map[r], param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(self.param_name_map[r], param_name) is not None:
                    return False
        return True
