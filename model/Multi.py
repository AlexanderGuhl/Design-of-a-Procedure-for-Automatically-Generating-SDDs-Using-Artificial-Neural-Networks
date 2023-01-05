import torch
import torch.nn as nn
from OctConvs import FirstOctConv, LastOctConv, InterOctConv
import random


class Multi_BN(nn.Module):
    """
    Multi network with batchnorm
    """
    def __init__(self, hparam):
        # set seed
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        super(Multi_BN, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        # set hyperparameters
        self.hparam = hparam
        self.kernel1 = int(hparam["SAMPLE_SIZE"] * hparam["KERNEL_FACTOR"])
        self.kernel2 = int(hparam["SAMPLE_SIZE"] * hparam["KERNEL_FACTOR"])
        self.kernel3 = int(hparam["SAMPLE_SIZE"] * hparam["KERNEL_FACTOR"])
        self.mp_kernel = hparam["MAXPOOL_KERNEL"]
        # set activation function of feature extraction network
        if hparam["ACTIVATION_FUNCTION"] == "CELU":
            self.act = nn.CELU()
        elif hparam["ACTIVATION_FUNCTION"] == "ReLU":
            self.act = nn.ReLU()
        elif hparam["ACTIVATION_FUNCTION"] == "Tanh":
            self.act = nn.Tanh()
        # set pooling factor and calculate needed stride and kernel size of maxpool
        self.p = 5      
        self.kernel_stride_mp3 = int((5 + hparam["SAMPLE_SIZE"] - 2 * self.mp_kernel - self.kernel1 - self.kernel2 - self.kernel3) / self.p)
        self.kernel_stride_mp5 = int((4 + hparam["SAMPLE_SIZE"] - self.mp_kernel - self.kernel1 - self.kernel2 - self.kernel3) / self.p)
        self.kernel_stride_mp6 = int((3 + hparam["SAMPLE_SIZE"] - self.kernel1 - self.kernel2 - self.kernel3) / self.p)
        
        # define left branch
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=(self.kernel1,), stride=(1,))
        self.mp1 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1)
        self.bn1 = nn.LazyBatchNorm1d()
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(self.kernel2,), stride=(1,))
        self.mp2 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1)
        self.bn2 = nn.LazyBatchNorm1d()
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(self.kernel3,), stride=(1,))
        # n/p = kernelsize und stride
        self.mp3 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp3, stride=self.kernel_stride_mp3)
        # define uniqe parts of middle branch
        self.bn3 = nn.LazyBatchNorm1d()
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(self.kernel2,), stride=(1,))
        self.mp4 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1)
        self.bn4 = nn.LazyBatchNorm1d()
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(self.kernel3,), stride=(1,))
        # n/p = kernelsize und stride
        self.mp5 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp5, stride=self.kernel_stride_mp5)
        # define unique parts of right branch
        self.bn5 = nn.LazyBatchNorm1d()
        self.conv6 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(self.kernel3,), stride=(1,))
        # n/p = kernelsize und stride
        self.mp6 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp6, stride=self.kernel_stride_mp6)
        self.bn6 = nn.LazyBatchNorm1d()
        # define global convolution with maxpool
        self.global_conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=(50,), stride=(1,))
        self.mp7 = nn.MaxPool1d(kernel_size=25, stride=1)
        # create MLP with ReLU
        self.MLP = nn.Sequential(
            nn.LazyLinear(1024),     
            nn.ReLU(),
            nn.LazyLinear(512),     
            nn.ReLU(),
            nn.LazyLinear(55-6)
        )

    def forward(self, x, e):
        # left branch
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.act(self.bn1(x))
        x1 = self.mp1(x)
        x1 = self.conv2(x1)
        x1 = self.act(self.bn2(x1))
        x1 = self.mp2(x1)
        x1 = self.conv3(x1)
        x1 = self.act(self.bn3(x1))
        x1 = self.mp3(x1)

        # middle branch
        x2 = self.conv4(x)
        x2 = self.act(self.bn4(x2))
        x3 = self.mp4(x2)
        x3 = self.conv5(x3)
        x3 = self.act(self.bn5(x3))
        x3 = self.mp5(x3)

        # right branch
        x4 = self.conv6(x2)
        x4 = self.act(self.bn6(x4))
        x4 = self.mp6(x4)

        x_out = torch.concat((x1, x3, x4), dim=1)
        x_out = torch.flatten(x_out, start_dim=1)

        x_out = self.global_conv(x_out.unsqueeze(dim=1))
        x_out = self.mp7(x_out)
        x_out = torch.flatten(x_out, start_dim=1)
        # MLP
        out = self.MLP(x_out)

        return out, e


class Multi(nn.Module):
    """
    basic multi without batchnorm
    """
    def __init__(self, hparam):
        # set seed
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        super(Multi, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        # set hyperparams
        self.hparam = hparam
        # set kernelsizes of convolutions
        self.kernel1 = int(hparam["SAMPLE_SIZE"] * 0.10)
        self.kernel2 = int(hparam["SAMPLE_SIZE"] * 0.1)
        self.kernel3 = int(hparam["SAMPLE_SIZE"] * 0.1)
        self.mp_kernel = hparam["MAXPOOL_KERNEL"]
        # set activation function
        if hparam["ACTIVATION_FUNCTION"] == "CELU":
            self.act = nn.CELU()
        elif hparam["ACTIVATION_FUNCTION"] == "ReLU":
            self.act = nn.ReLU()
        elif hparam["ACTIVATION_FUNCTION"] == "Tanh":
            self.act = nn.Tanh()

        self.mlp_act = nn.CELU()
        # set pooling factor and calculate needed stride and kernel size of maxpool
        self.p = 5
        self.kernel_stride_mp3 = int((5 + hparam["SAMPLE_SIZE"] - 2*self.mp_kernel - self.kernel1 - self.kernel2 - self.kernel3)/self.p)
        self.kernel_stride_mp5 = int((4 + hparam["SAMPLE_SIZE"] - self.mp_kernel - self.kernel1 - int(self.kernel2/2) - int(self.kernel3/4))/self.p)
        self.kernel_stride_mp6 = int((3 + hparam["SAMPLE_SIZE"] - self.kernel1 - int(self.kernel2/2) - int(self.kernel3/4))/self.p)
        
        # define left branch
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=(self.kernel1,), stride=(1,))
        self.mp1 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(self.kernel2,), stride=(1,))
        self.mp2 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(self.kernel3,), stride=(1,))
        # n/p = kernelsize und stride
        self.mp3 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp3, stride=self.kernel_stride_mp3)
        
        # define unique parts of middle branch
        self.conv4 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(int(self.kernel2/2),), stride=(1,))
        self.mp4 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(self.kernel3,), stride=(1,))
        # n/p = kernelsize und stride
        self.mp5 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp5, stride=self.kernel_stride_mp5)
        
        # define unique parts of right branch
        self.conv6 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(int(self.kernel3/4),), stride=(1,))
        # n/p = kernelsize und stride
        self.mp6 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp6, stride=self.kernel_stride_mp6)

        self.global_conv = nn.Conv1d(in_channels=96, out_channels=512, kernel_size=(3,), stride=(1,))
        self.mp7 = nn.MaxPool1d(kernel_size=25, stride=1)
        # define MLP
        self.lin1 = nn.LazyLinear(1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3 = nn.Linear(512, 55-6)

    def forward(self, x, e):
        # left branch
        x = self.conv1(x.unsqueeze(dim=1))
        x = self.act(x)
        x1 = self.mp1(x)
        x1 = self.conv2(x1)
        x1 = self.act(x1)
        x1 = self.mp2(x1)
        x1 = self.conv3(x1)
        x1 = self.act(x1)
        x1 = self.mp3(x1)

        # middle branch
        x2 = self.conv4(x)
        x2 = self.act(x2)
        x3 = self.mp4(x2)
        x3 = self.conv5(x3)
        x3 = self.act(x3)
        x3 = self.mp5(x3)

        # right branch
        x4 = self.conv6(x2)
        x4 = self.act(x4)
        x4 = self.mp6(x4)

        x_out = torch.concat((x1, x3, x4), dim=1)
        x_out = torch.flatten(x_out, start_dim=1)

        x_out = self.global_conv(x_out.unsqueeze(dim=1))
        x_out = self.mp7(x_out)
        x_out = torch.flatten(x_out, start_dim=1)

        # MLP
        out = self.lin1(x_out)
        out = self.mlp_act(out)
        out = self.lin2(out)
        out = self.mlp_act(out)
        out = self.lin3(out)

        return out, e


class MultiOct_BN(nn.Module):
    """
    Multi network with octave convolutions
    """
    def __init__(self, hparam):
        random.seed()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])
        super(MultiOct_BN, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        random.seed(hparam["MAN_SEED"])

        self.hparam = hparam
        self.mp_kernel = hparam["MAXPOOL_KERNEL"]
        # set activation function of feature extraction network
        if hparam["ACTIVATION_FUNCTION"] == "CELU":
            self.act = nn.CELU()
        elif hparam["ACTIVATION_FUNCTION"] == "ReLU":
            self.act = nn.ReLU()
        elif hparam["ACTIVATION_FUNCTION"] == "Tanh":
            self.act = nn.Tanh()

        self.mlp_act = nn.ReLU()        
        #set pooling factor and calcuate stride and kernelsize for maxpool at end of branches
        self.p = 5
        self.kernel_stride_mp = int(hparam["SAMPLE_SIZE"]/self.p)
        # define legt branch
        self.conv1 = FirstOctConv(hparam, 1, 8)
        self.mp1 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1, padding=int((self.mp_kernel-1)/2))
        self.bn1 = nn.LazyBatchNorm1d()
        self.conv2 = InterOctConv(hparam, 8, 16)
        self.mp2 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1,padding=int((self.mp_kernel-1)/2))
        self.bn2 = nn.LazyBatchNorm1d()
        self.conv3 = LastOctConv(hparam, 16,  32)
        # n/p = kernelsize und stride
        self.mp3 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp, stride=self.kernel_stride_mp)

        # define uniqe parts of middle branch
        self.bn3 = nn.LazyBatchNorm1d()
        self.conv4 = InterOctConv(hparam, 8, 16)
        self.mp4 = nn.MaxPool1d(kernel_size=self.mp_kernel, stride=1, padding=int((self.mp_kernel-1)/2))
        self.bn4 = nn.LazyBatchNorm1d()
        self.conv5 = LastOctConv(hparam, 16, 32)
        # n/p = kernelsize und stride
        self.mp5 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp, stride=self.kernel_stride_mp)
        
        # define unique parts of right branch
        self.bn5 = nn.LazyBatchNorm1d()
        self.conv6 = LastOctConv(hparam, 16, 32)
        # n/p = kernelsize und stride
        self.mp6 = nn.MaxPool1d(kernel_size=self.kernel_stride_mp, stride=self.kernel_stride_mp)
        # define additional batchnorm, additional with respect to multi_bn
        self.bn6 = nn.LazyBatchNorm1d()
        self.bn7 = nn.LazyBatchNorm1d()
        self.bn8 = nn.LazyBatchNorm1d()
        self.bn9 = nn.LazyBatchNorm1d()

        self.global_conv = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=(50,), stride=(1,))
        self.mp7 = nn.MaxPool1d(kernel_size=25, stride=1)
        # define MLP
        self.MLP = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 55-6),
        )

    def forward(self, x, e):
        # left branch
        x_h, x_l = self.conv1(x.unsqueeze(dim=1))
        x_h = self.act(self.bn1(x_h))
        x_l = self.act(self.bn2(x_l))

        x1_h = self.mp1(x_h)
        x1_l = self.mp1(x_l)

        x1_h, x1_l = self.conv2(x1_h, x1_l)
        x1_h = self.act(self.bn3(x1_h))
        x1_l = self.act(self.bn4(x1_l))

        x1_h = self.mp2(x1_h)
        x1_l = self.mp2(x1_l)

        x1 = self.conv3(x1_h, x1_l)
        x1 = self.act(self.bn5(x1))
        x1 = self.mp3(x1)

        # middle branch
        x2_h, x2_l = self.conv4(x_h, x_l)
        x2_h = self.act(self.bn6(x2_h))
        x2_l = self.act(self.bn7(x2_l)) # mehr unterschiedliche BNs

        x3_h = self.mp4(x2_h)
        x3_l = self.mp4(x2_l)

        x3 = self.conv5(x3_h, x3_l)
        x3 = self.act(self.bn8(x3))
        x3 = self.mp5(x3)

        # right branch
        x4 = self.conv6(x2_h, x2_l)
        x4 = self.act(self.bn9(x4))
        x4 = self.mp6(x4)

        x_out = torch.concat((x1, x3, x4), dim=1)
        x_out = torch.flatten(x_out, start_dim=1)
        x_out = self.global_conv(x_out.unsqueeze(dim=1))
        x_out = self.mp7(x_out)
        x_out = torch.flatten(x_out, start_dim=1)

        # MLP
        out = self.MLP(x_out)

        return out, e


