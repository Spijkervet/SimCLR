import torchvision


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT
                                                if pretrained else None),
        "resnet50": torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT
                                                if pretrained else None),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
