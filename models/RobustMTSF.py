import numpy as np
# import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layer.Embed import DataEmbedding
from layer.RobustMTSF_Block import GraphBlock, Predict
from layer.Conv_Blocks import Inception_Block_V1
import torch
import torch.nn as nn



import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size = 25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


device = torch.device("cuda:0")

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]





class ScaleBlock(nn.Module):
    def __init__(self, configs):
        super(ScaleBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.norm = nn.LayerNorm(configs.d_model)
        self.gelu = nn.GELU()
        self.gconv = nn.ModuleList()

        self.weight = nn.Parameter(torch.randn(configs.batch_size, configs.top_k), requires_grad=True)
        self.lay = nn.Linear(configs.d_model ,configs.d_model)

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )
        for i in range(self.k):
            self.gconv.append(
                GraphBlock(configs.c_out , configs.d_model , configs.conv_channel, configs.skip_channel,
                        configs.gcn_depth , configs.dropout, configs.propalpha ,configs.seq_len,
                           configs.node_dim))


    def forward(self, x):
        B, T, N = x.size()
        scale_list, scale_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            scale = scale_list[i]
            #Gconv
            x = self.gconv[i](x,scale)
            # paddng
            if (self.seq_len) % scale != 0:
                length = (((self.seq_len) // scale) + 1) * scale
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // scale, scale, N).permute(0, 3, 1, 2).contiguous()
            out1 = out
            out = self.conv(out)
            out += out1
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out = self.norm(out)
            out = out[:, :self.seq_len, :]
            res.append(out)

        res = torch.stack(res,
                          dim=-1)



        # adaptive aggregation
        scale_weight = self.weight
        scale_weight = F.softmax(scale_weight, dim=1)
        scale_weight = scale_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)

        res = torch.sum(res * scale_weight, -1)

        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.model = nn.ModuleList([ScaleBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.layer_norm1 = nn.LayerNorm(configs.enc_in)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)
        self.seq2pred = Predict(configs.individual ,configs.c_out,
                                configs.seq_len, configs.pred_len, configs.dropout)
        self.revin = RevIN(self.configs.enc_in)
        self.decom = series_decomp()
        self.Linear1 = nn.Sequential(
            nn.Linear(configs.d_model,configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff,configs.d_model)

        )
        self.Linear2 = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model)

        )
        self.glue = nn.GELU()

        self.Linear3 = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.c_out)

        )
        self.Linear4 = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.c_out)

        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        x_enc = self.revin(x_enc,'norm')
        B , T , N = x_enc.shape
        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        x_seasonal , x_trend = self.decom(enc_out)
        x_seasonal = self.Linear1(x_seasonal)
        x_trend = self.Linear2(x_trend)
        enc_out = x_trend + x_seasonal + enc_out

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)


        dec_out = self.seq2pred(dec_out.transpose(1, 2)).transpose(1, 2)
        dec_trend, dec_seasonal = self.decom(dec_out)
        dec_trend = self.Linear3(dec_trend)
        dec_seasonal = self.Linear4(dec_seasonal)
        dec_out1 = dec_trend + dec_seasonal
        dec_out = dec_out  + dec_out1*0.01

        dec_out = self.revin(dec_out,'denorm')
        return dec_out[:, -self.pred_len:, :]


