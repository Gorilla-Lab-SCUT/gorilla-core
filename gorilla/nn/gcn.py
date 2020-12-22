import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" \
               + str(self.in_features) + " -> " \
               + str(self.out_features) + ")"


class GCN(nn.Sequential):
    def __init__(self,
                 channels: List[int],
                 dropout: float=0.0):
        r"""Author: liang.zhihao
        Graph Convolution Block

        Args:
            channels (List[int]):  The num of features of each layer (including input layer and output layer).
            dropout (float, optional): Dropout ratio. Defaults to 0.0.
        """
        super(GCN, self).__init__()
        assert len(channels) >= 2
        self.num_layers = len(channels) - 1
        for idx, (in_features, out_features) in enumerate(zip(channels[:-1], channels[1:])):
            idx = idx + 1
            self.add_module("gc{}".format(idx), GraphConvolution(in_features, out_features))
            # explict the last layer
            if idx != self.num_layers:
                self.add_module("ReLU{}".format(idx), nn.ReLU(inplace=True))
                self.add_module("dropout{}".format(idx), nn.Dropout(dropout))

