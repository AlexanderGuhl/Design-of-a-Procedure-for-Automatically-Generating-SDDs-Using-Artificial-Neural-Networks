import torch
import torch.nn as nn


class Oct_Conv(nn.Module):
    def __init__(self, hparam):
        """
        convs need to be sized so that the output of the low filter is half the length of the high filter
        OctConv as one module
        """
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        super(Oct_Conv, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        self.hparam = hparam
        self.alpha = self.hparam["ALPHA"]
        # first oct conv
        self.init_filters = 4
        self.first_h = nn.Conv1d(in_channels=1, out_channels=int(self.alpha*self.init_filters), kernel_size=(3,),padding=1,dilation=(1,))
        self.first_l = self.lf = nn.Conv1d(in_channels=1, out_channels=int((1-self.alpha)*self.init_filters), kernel_size=(3,),padding=1,dilation=(1,))
        self.first_avg_pool = nn.AvgPool1d(kernel_size=(2,), stride=2)
        # intermediate oct conv
        # scale _factor needs to be power of 2, as it defines an octave
        self.inter_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # avg_pool has same kernel size as stride
        self.inter_avg_pool = nn.AvgPool1d(kernel_size=(2,), stride=2)
        self.inter_l_h = nn.Conv1d(in_channels=int((1-self.alpha)*self.init_filters), out_channels=int((1-self.alpha)*self.init_filters), kernel_size=(3,), padding=1)
        self.inter_h_h = nn.Conv1d(in_channels=int(self.alpha*self.init_filters), out_channels=int((1-self.alpha)*self.init_filters), kernel_size=(3,), padding=1)
        self.inter_h_l = nn.Conv1d(in_channels=int(self.alpha*self.init_filters), out_channels=int(self.alpha*self.init_filters), kernel_size=(3,), padding=1)
        self.inter_l_l = nn.Conv1d(in_channels=int((1-self.alpha)*self.init_filters), out_channels=int(self.alpha*self.init_filters), kernel_size=(3,), padding=1)
        # final oct conv
        self.final_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_h = nn.Conv1d(in_channels=int((1-self.alpha)*self.init_filters), out_channels=1, kernel_size=(3,), padding=1, dilation=(1,))
        self.final_l = nn.Conv1d(in_channels=int(self.alpha*self.init_filters), out_channels=1, kernel_size=(3,), padding=1, dilation=(1,))

        if hparam["ACTIVATION_FUNCTION"] == "CELU":
            self.act = nn.CELU()
        elif hparam["ACTIVATION_FUNCTION"] == "ReLU6":
            self.act = nn.ReLU6()
        elif hparam["ACTIVATION_FUNCTION"] == "Tanh":
            self.act = nn.Tanh()

        self.bn1 = nn.LazyBatchNorm1d()
        self.bn2 = nn.LazyBatchNorm1d()
        self.bn3 = nn.LazyBatchNorm1d()
        self.bn4 = nn.LazyBatchNorm1d()
        self.bn5 = nn.LazyBatchNorm1d()
        self.bn6 = nn.LazyBatchNorm1d()

    def forward(self, x, e):
        # first high filter
        x_h = self.first_h(x.unsqueeze(dim=1))
        x_h = self.bn1(x_h)
        x_h = self.act(x_h)
        # first low filter
        x_l = self.first_avg_pool(x.unsqueeze(dim=1))
        x_l = self.first_l(x_l)
        x_l = self.bn2(x_l)
        x_l = self.act(x_l)

        # intermediate oct conv, high part
        x_h_h = self.inter_h_h(x_h)
        x_l_h = self.inter_l_h(x_l)
        x_l_h = self.inter_upsample(x_l_h)
        x_h_inter = x_l_h + x_h_h
        x_h_inter = self.bn3(x_h_inter)
        x_h_inter = self.act(x_h_inter)
        # intermediate oct conv, low part
        # avg pool reduces length of x_h to length of x_l
        x_h_l = self.inter_avg_pool(x_h)
        x_h_l = self.inter_h_l(x_h_l)
        x_l_l = self.inter_l_l(x_l)
        x_l_inter = x_h_l + x_l_l
        x_l_inter = self.bn4(x_l_inter)
        x_l_inter = self.act(x_l_inter)
        # final oct conv
        x_h_f = self.final_h(x_h_inter)
        x_h_f = self.bn5(x_h_f)
        x_l_f = self.final_l(x_l_inter)
        x_l_f = self.final_upsample(x_l_f)
        x_l_f = self.bn6(x_l_f)
        #x_out = self.act_final_h(x_h_f) + self.act_final_l(x_l_f)
        return self.act(x_h_f), self.act(x_l_f), e


class FirstOctConv(nn.Module):
    def __init__(self, hparam, in_fil, out_fil):
        """
        convs need to be sized so that the output of the low filter is half the length of the high filter
        OctConv as one module
        """
        torch.manual_seed(hparam["MAN_SEED"])
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        super(FirstOctConv, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])

        self.alpha = hparam["ALPHA"]
        self.in_fil = in_fil
        self.out_fil = out_fil
        self.kernel = int(hparam["SAMPLE_SIZE"] * 0.10)-1

        self.h_h = nn.Conv1d(in_channels=self.in_fil, out_channels=int(self.alpha * self.out_fil),
                             kernel_size=(self.kernel,), padding="same", padding_mode="reflect", dilation=(1,))
        self.h_l = self.lf = nn.Conv1d(in_channels=self.in_fil, out_channels=int((1 - self.alpha) * self.out_fil),
                                           kernel_size=(self.kernel,), padding="same", padding_mode="reflect", dilation=(1,))
        self.avg_pool = nn.AvgPool1d(kernel_size=(2,), stride=2)

    def forward(self, x):
        x_h = self.h_h(x)
        x_l = self.avg_pool(x)
        x_l = self.h_l(x_l)
        return x_h, x_l


class LastOctConv(nn.Module):
    def __init__(self, hparam, in_fil, out_fil):
        """
        convs need to be sized so that the output of the low filter is half the length of the high filter
        OctConv as one module
        """
        torch.manual_seed(hparam["MAN_SEED"])
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        super(LastOctConv, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])

        self.alpha = hparam["ALPHA"]
        self.in_fil = in_fil
        self.out_fil = out_fil
        self.kernel = int(hparam["SAMPLE_SIZE"] * 0.10)-1

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.final_h = nn.Conv1d(in_channels=int((1 - self.alpha) * self.in_fil), out_channels=self.out_fil,
                                 kernel_size=(self.kernel,), padding="same", padding_mode="reflect")
        self.final_l = nn.Conv1d(in_channels=int(self.alpha * self.in_fil), out_channels=self.out_fil,
                                 kernel_size=(self.kernel,), padding="same", padding_mode="reflect")

    def forward(self, x_h, x_l):
        x_h = self.final_h(x_h)
        x_l = self.final_l(x_l)
        x_l = self.upsample(x_l)
        x_out = x_h + x_l
        return x_out


class InterOctConv(nn.Module):
    def __init__(self, hparam, in_fil, out_fil):
        """
        convs need to be sized so that the output of the low filter is half the length of the high filter
        OctConv as one module
        """
        torch.manual_seed(hparam["MAN_SEED"])
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])
        super(InterOctConv, self).__init__()
        torch.manual_seed(hparam["MAN_SEED"])
        torch.manual_seed(hparam["MAN_SEED"])
        torch.cuda.manual_seed(hparam["MAN_SEED"])

        self.alpha = hparam["ALPHA"]
        self.in_fil = in_fil
        self.out_fil = out_fil
        self.kernel = int(hparam["SAMPLE_SIZE"] * 0.10)-1

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.avg_pool = nn.AvgPool1d(kernel_size=(2,), stride=2)
        self.l_h = nn.Conv1d(in_channels=int((1-self.alpha)*self.in_fil), out_channels=int((1-self.alpha)*self.out_fil),
                             kernel_size=(self.kernel,), padding="same", padding_mode="reflect")
        self.h_h = nn.Conv1d(in_channels=int(self.alpha*self.in_fil), out_channels=int((1-self.alpha)*self.out_fil),
                             kernel_size=(self.kernel,), padding="same", padding_mode="reflect")
        self.h_l = nn.Conv1d(in_channels=int(self.alpha*self.in_fil), out_channels=int(self.alpha*self.out_fil),
                             kernel_size=(self.kernel,), padding="same", padding_mode="reflect")
        self.l_l = nn.Conv1d(in_channels=int((1-self.alpha)*self.in_fil), out_channels=int(self.alpha*self.out_fil),
                             kernel_size=(self.kernel,), padding="same", padding_mode="reflect")

    def forward(self, x_h, x_l):
        # low frequency
        x_h_l = self.avg_pool(x_h)
        x_h_l = self.h_l(x_h_l)
        x_l_l = self.l_l(x_l)

        x_l_out = x_l_l + x_h_l
        # high frequency
        x_l_h = self.l_h(x_l)
        x_l_h = self.upsample(x_l_h)
        x_h_h = self.h_h(x_h)
        x_h_out = x_h_h + x_l_h
        return x_h_out, x_l_out
