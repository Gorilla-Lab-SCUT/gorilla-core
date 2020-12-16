import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
import math

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with zero-padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        r"""
        The number of channels in the network:
        input: inplanes --> planes --> planes: output
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        if self.downsample is not None:
            x = self.downsample(x)

        out = residual + x
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.relu(self.bn2(residual))
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.downsample is not None:
            x = self.downsample(x)

        out = residual + x
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        r"""
        Parameters:
        ----------
        block: a class inherited from nn.Module
            The basic network block
        num_blocks: list
            The number of basic blocks in four main convolution layers of ResNet
        num_classes: int, optional
            The number of classes to be classified
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        # the shape of input image is [224, 224, 3]
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)  # [112, 112, 64]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1)  # [56, 56, 64]
        self.layer1 = self._make_layer(block, 64,
                                       num_blocks[0])  # [56, 56, 64]
        self.layer2 = self._make_layer(block, 128, num_blocks[1],
                                       stride=2)  # [28, 28, 128]
        self.layer3 = self._make_layer(block, 256, num_blocks[2],
                                       stride=2)  # [14, 14, 256]
        self.layer4 = self._make_layer(block, 512, num_blocks[3],
                                       stride=2)  # [7, 7, 512]
        self.avgpool = nn.AvgPool2d(7)  # [1, 1, 512]
        #         self.fc = nn.Linear(512 * block.expansion, num_classes) # [1000]

        # initial weight and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_block, stride=1):
        r"""
        Make a layer that contain some residual blocks.
        Parameters
        ----------
        block: a class inherited from nn.Module
            The basic network block
        planes: int
            The number of output planes of the first network unit of a block
        num_block: int
            The number of blocks
        stride: int, optional
            If the stride is not equal to 1, then take a downsampling with the stride
        Return
        ------
        layer: nn.Sequential
            The network architecture of a layer
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        blocks = []
        # The downsampling operation just appear in the first block of a layer,
        # to reduce the size of image adapt the number of channel to the output one
        blocks.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # renew current number of input planes after going through a block
        for i in range(1, num_block):
            # There is self.inplanes == planes * block.expansion in the latter block, so don't need to renew it again
            blocks.append(block(self.inplanes, planes))

        layer = nn.Sequential(*blocks)

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # [batch_size, 1, 1, 512]
        x = x.view(x.size(0), -1)  # [batch_size, 512]
        #         x = self.fc(x)

        return x


def resnet18(args, **kwargs):
    r"""
    Constructs a ResNet-18 model
    Parameters
    ----------
    pretrained: bool
        If True, return a model pre-trained on ImageNet
    Return
    ------
    model: nn.Sequential
        A ResNet model
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    pretrained_dict = {}
    if args.pretrained:
        print("Load the ImageNet pretrained model")
        pretrained_dict = model_zoo.load_url(model_urls["resnet18"])
        model_dict = model.state_dict()
        # The length of model_dict is 120, and the length of pretrain_dict is 102, but the difference is just that every BN layer lack a parameter named "num_batches_tracked", and the last fc layer"s weight and bias(17+1[metadata]+2)
        pretrained_dict_temp = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)

    return model


def resnet34(args, **kwargs):
    r"""
    Constructs a ResNet-34 model
    Parameters
    ----------
    pretrained: bool
        If True, return a model pre-trained on ImageNet
    Return
    ------
    model: nn.Sequential
        A ResNet model
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    pretrained_dict = {}
    if args.pretrained:
        print("Load the ImageNet pretrained model")
        pretrained_dict = model_zoo.load_url(model_urls["resnet34"])
        model_dict = model.state_dict()
        pretrained_dict_temp = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)

    return model


def resnet50(args, **kwargs):
    r"""Constructs a ResNet-50 model.
    Parameters
    ----------
    pretrained: bool
        If True, return a model pre-trained on ImageNet
    Return
    ------
    model: nn.Sequential
        A ResNet model
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if args.pretrained:
        print("Load the ImageNet pretrained model")
        pretrained_dict = model_zoo.load_url(model_urls["resnet50"])
        model_dict = model.state_dict()
        pretrained_dict_temp = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)

    return model


def resnet101(args, **kwargs):
    r"""
    Constructs a ResNet-101 model
    Parameters
    ----------
    pretrained: bool
        If True, return a model pre-trained on ImageNet
    Return
    ------
    model: nn.Sequential
        A ResNet model
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if args.pretrained:
        print("Load the ImageNet pretrained model")
        pretrained_dict = model_zoo.load_url(model_urls["resnet101"])
        model_dict = model.state_dict()
        pretrained_dict_temp = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)

    return model


def resnet152(args, **kwargs):
    r"""
    Constructs a ResNet-152 model
    Parameters
    ----------
    pretrained: bool
        If True, return a model pre-trained on ImageNet
    Return
    ------
    model: nn.Sequential
        A ResNet model
    """
    pass

    return 


def resnet(args, **kwargs):
    print("==> Creating model '{}'".format(args.arch))
    if args.arch == "resnet18":
        return resnet18(args)
    elif args.arch == "resnet34":
        return resnet34(args)
    elif args.arch == "resnet50":
        return resnet50(args)
    elif args.arch == "resnet101":
        return resnet101(args)
    elif args.arch == "resnet152":
        return resnet152(args)
    else:
        raise ValueError("Unrecognized model architecture {}".format(args.arch))
