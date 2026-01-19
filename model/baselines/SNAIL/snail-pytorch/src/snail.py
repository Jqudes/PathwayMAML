import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from resnet_blocks import *
from blocks import *

class SnailFewShot(nn.Module):

    def __init__(self, N, K, feature_dim, use_cuda=True):
        super().__init__()
        self.N = N
        self.K = K
        self.use_cuda = use_cuda

        # “raw feature_dim → embedding_dim” 매핑
        embedding_dim = 1024
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        num_channels = embedding_dim + N
        # AttentionBlock
        self.attention1 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        # TCBlock
        self.tc1 = TCBlock(num_channels, N * K + 1, 1024)
        num_channels += int(math.ceil(math.log(N * K + 1, 2))) * 1024
        # AttentionBlock
        self.attention2 = AttentionBlock(num_channels, 2048, 1024)
        num_channels += 1024
        # TCBlock
        self.tc2 = TCBlock(num_channels, N * K + 1, 1024)
        num_channels += int(math.ceil(math.log(N * K + 1, 2))) * 1024
        # AttentionBlock
        self.attention3 = AttentionBlock(num_channels, 4096, 2048)
        num_channels += 2048
        # N-way 분류를 위한 FC
        self.fc = nn.Linear(num_channels, N)

    def forward(self, input, labels):
        x = self.encoder(input)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]
        if self.use_cuda:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        else:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.N * self.K + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x
