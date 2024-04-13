import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

logger = logging.getLogger(__name__)


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DcWideResNet(nn.Module):
    '''
        overwrite by dhxy in Sep 10 2023,inspired by method of parameters decompositon in
        FedMatch
    '''
    def __init__(self, args, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(DcWideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        self.args = args
        self.num_classes = num_classes
        self.depth=depth
        self.widen_factor=widen_factor
        self.drop_rate=drop_rate
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        self.sigma = [copy.deepcopy(item) for item in self.parameters()]
        self.psi = [nn.Parameter(item * 0.2) for item in self.sigma]
    def __deepcopy__(self, memo):
        # 创建一个新的实例
        new_instance = self.__class__(
            args=copy.copy(self.args),
            depth=self.depth,
            widen_factor=self.widen_factor,
            drop_rate=self.drop_rate,
            num_classes=self.num_classes
        )

        # 复制模型的状态字典
        new_instance.load_state_dict(copy.deepcopy(self.state_dict(), memo))

        return new_instance
    def forward_x(self, x):

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        return self.fc(out)

    def forward(self, inputs, update_para = False):
        if update_para:
            self.update_theta()
        if not isinstance(inputs, list):
            return self.forward_x(inputs)
        outputs = []
        for input in inputs:
            outputs.append(self.forward_x(input))
        return outputs
    def get_sigma(self):
        return self.sigma

    def get_psi(self):
        return self.psi

    def update_theta(self):
        idx = 0
        for para in self.parameters():
            para.data = self.psi[idx] + self.sigma[idx]
            idx += 1



def build_dcwideresnet(args, depth, widen_factor, dropout, num_classes):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return DcWideResNet(args=args,
                        depth=depth,
                        widen_factor=widen_factor,
                        drop_rate=dropout,
                        num_classes=num_classes)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma = [nn.Parameter(torch.ones(3,3)*3.14)]
        self.psi = [nn.Parameter(torch.ones(3,3)*2.14)]
        self.sigma = self.parameters()
        self.k = nn.ParameterList([nn.Parameter(torch.zeros(3,3))])
    def niu(self):
        self.g = nn.Parameter(torch.ones(4,4))


if __name__ == "__main__":
    net = DcWideResNet(None, 10)
    print(net.parameters())
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.999},
        {'params': [p for n, p in net.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    from torch import optim
    sigma = [copy.deepcopy(item) for item in net.parameters()]
    psi = [nn.Parameter(item * 0.2) for item in sigma]
    idx = 0
    for para in net.parameters():
        para.data = psi[idx] + sigma[idx]
        idx += 1
    optimizer_s = optim.SGD(sigma, lr=0.003)
    optimizer_p = optim.SGD(psi, lr=0.003)
    optimizer = optim.SGD(net.parameters(), lr=0.003)

    print(net)
    n = Net()
    n.niu()
    print(n)
