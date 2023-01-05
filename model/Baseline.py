import torch
import torch.nn as nn
import random


class CompressNet(nn.Module):
    """
    Model to compress a longer time series.
    Used to compress time series of UR10e to length of UR3, while keeping patterns.
    """
    def __init__(self, seed, output_size):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        super(CompressNet, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        y = self.pool(x.unsqueeze(dim=0))
        y = y.squeeze(dim=0)
        return y


class Baseline(nn.Module):
    """
    Baseline CNN with MLP and three Convolutions and Maxpoolings
    Used as reference for benchmark
    """
    def __init__(self, hparam):
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        super(Baseline, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        self.hparam = hparam
        self.filter = int(hparam["SAMPLE_SIZE"] * hparam["KERNEL_FACTOR"])
        self.mlp_act = nn.ReLU()
        if hparam["ACTIVATION_FUNCTION"] == "CELU":
            self.act = nn.CELU()
        elif hparam["ACTIVATION_FUNCTION"] == "ReLU":
            self.act = nn.ReLU()
        elif hparam["ACTIVATION_FUNCTION"] == "Tanh":
            self.act = nn.Tanh()

        self.mp_kernel = hparam["MAXPOOL_KERNEL"]

        self.first_layer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=(self.filter,), stride=(1,)),
            nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1),
            self.act
        )
        self.second_layer = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=(self.filter,), stride=(1,)),
            nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1),
            self.act
        )
        self.third_layer = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=(self.filter,), stride=(1,)),
            nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1),
            self.act
        )
        # create MLP
        self.MLP = nn.Sequential(
            nn.LazyLinear(2048),
            self.mlp_act,
            nn.LazyLinear(1024),
            self.mlp_act,
            nn.LazyLinear(55)
        )

    def forward(self, x, e):
        x = self.first_layer(x.unsqueeze(dim=1))
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = torch.flatten(x, start_dim=1)
        out = self.MLP(x)
        return out, e