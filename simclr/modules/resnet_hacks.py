import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet


def modify_resnet_model(model, *, cifar_stem=True, v1=True):
    """Modifies some layers of a given torchvision resnet model to
    match the one used for the CIFAR-10 experiments in the SimCLR paper.

    Parameters
    ----------
    model : ResNet
        Instance of a torchvision ResNet model.
    cifar_stem : bool
        If True, adapt the network stem to handle the smaller CIFAR images, following
        the SimCLR paper. Specifically, use a smaller 3x3 kernel and 1x1 strides in the
        first convolution and remove the max pooling layer.
    v1 : bool
        If True, modify some convolution layers to follow the resnet specification of the
        original paper (v1). torchvision's resnet is v1.5 so to revert to v1 we switch the
        strides between the first 1x1 and following 3x3 convolution on the first bottleneck
        block of each of the 2nd, 3rd and 4th layers.

    Returns
    -------
    Modified ResNet model.
    """
    assert isinstance(model, ResNet), "model must be a ResNet instance"
    if cifar_stem:
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        model.conv1 = conv1
        model.maxpool = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, "layer{}".format(l))
            block = list(layer.children())[0]
            if isinstance(block, Bottleneck):
                assert block.conv1.kernel_size == (1, 1) and block.conv1.stride == (
                    1,
                    1,
                )
                assert block.conv2.kernel_size == (3, 3) and block.conv2.stride == (
                    2,
                    2,
                )
                assert block.conv2.dilation == (
                    1,
                    1,
                ), "Currently, only models with dilation=1 are supported"
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model


# You can check the architecture with the following snippet
# from torchsummary import summary
# from torchvision.models import resnet50
# summary(modify_resnet_model(resnet50(num_classes=10)), (3, 32, 32))
