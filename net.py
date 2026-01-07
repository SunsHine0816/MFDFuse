import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torch.autograd import Variable
import numbers
import kornia as k

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

# from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
#
# from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
# from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
#
# from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
# from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined



def drop_path(x, drop_prob: float = 0., training: bool = False):
    '''
    作为正则化手段加入网络，但是会增加网络训练的难度。尤其是在NAS问题中，如果设置的drop_prob过高，模型甚至有可能不收敛。
    '''
    # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
    if drop_prob == 0. or not training:
        return x
    # 保持率
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # 0-1之间的均匀分布
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    # 下取整从而确定保存哪些样本 总共有batch个数
    random_tensor.floor_()
    # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    output = x.div(keep_prob) * random_tensor
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CSSMBase(nn.Module):
    def __init__(self,
                 dim,
                 bias=False,
                 ):
        super(CSSMBase, self).__init__()
        self.proj = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dconv = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=bias)
        self.mamba_block = MambaBlock(dim)

    def forward(self, x):
        with torch.cuda.device(x.device):
            x_proj = self.proj(x)
            x_proj = self.dconv(x_proj)
            x_proj = x_proj.flatten(2).transpose(1, 2)
            x_proj = self.mamba_block(x_proj)
            x_proj = x_proj.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        return x_proj

class Mlp(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Conv2d(dim, dim*ffn_expansion_factor, kernel_size=3, stride=1, padding=1, bias=True)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
# -------------------------------------------------------------------------------------------------
# Restormer中的部分详细实现



# 详细可见Restormer论文
# class BaseFeatureExtraction(nn.Module):
#     def __init__(self,
#                  dim,
#                  ffn_expansion_factor,
#                  img_hw=128*128,
#                  bias=False,
#                  LayerNorm_type='WithBias'):
#         super(BaseFeatureExtraction, self).__init__()
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         # self.sssm = SSSM(dim=dim, img_hw=img_hw)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.cssm = CSSM(dim=dim)
#         self.ffn =Mlp(dim, ffn_expansion_factor)
#         self.dconv_one = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1, bias=bias)
#         self.dconv_two = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=bias)
#
#     def forward(self, x):
#         with torch.cuda.device(x.device):
#             x_cssm = x + self.cssm(self.norm2(x))
#             x_cssm = x_cssm + self.dconv_two(self.dconv_one(x_cssm))
#             x_ffn = x_cssm + self.ffn(self.norm1(x_cssm))
#
#         return x_ffn

#两个卷积层
class HighFrequencyBlock_1(nn.Module):
    def __init__(self, inp, oup):
        super(HighFrequencyBlock_1, self).__init__()
        self.reduction_conv = nn.Conv2d(inp, inp // 2, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(inp // 2)
        self.relu = nn.ReLU6(inplace=True)

        self.expansion_conv = nn.Conv2d(inp // 2, inp, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inp)
        self.relu2 = nn.ReLU6(inplace=True)

        self.basic_conv = nn.Conv2d(inp * 2, inp, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(inp)
        self.relu3 = nn.ReLU6(inplace=True)
        self.basic_conv2 = nn.Conv2d(inp, inp, kernel_size=3, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(inp)
        self.relu4 = nn.ReLU6(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.atten = nn.Sequential(
            nn.Conv2d(inp, inp // 2, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // 2, inp, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.basic_conv3 = nn.Conv2d(inp, oup, kernel_size=3, padding=1, bias=False)
        # self.norm5 = nn.BatchNorm2d(oup)
        self.relu5 = nn.ReLU6(inplace=True)
        self.norm5 = nn.Conv2d(inp, oup, 1, bias=False)
        self.ca = ChannelAttention(inp)

    def forward(self, x):
        x1 = self.relu(self.norm1(self.reduction_conv(x)))
        x1 = self.relu2(self.norm2(self.expansion_conv(x1)))
        x1 = torch.cat((x1, x), dim=1)
        x1 = self.relu3(self.norm3(self.basic_conv(x1)))
        x1 = self.relu4(self.norm4(self.basic_conv2(x1)))

        x2 = self.avg_pool(x1)
        x2 = self.atten(x2)
        # x2 = self.ca(x1)
        x2 = x1 * x2 + x1

        x2 = self.relu5(self.basic_conv3(x2))
        x2 = self.norm5(x2)

        return x2 + x1


class HighFrequencyBlock_2(nn.Module):
    def __init__(self, inp, oup):
        super(HighFrequencyBlock_2, self).__init__()
        self.cov_high = nn.Sequential(
            nn.Conv2d(inp, inp, 3, padding=1),
            nn.BatchNorm2d(inp),
            # nn.PReLU(),
            nn.Tanh(),
        )
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(inp, inp // 2, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // 2, oup, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.conv_1 = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        # self.norm =nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)
        self.down_c = nn.Conv2d(oup, oup, 1, bias=False)
        self.ca = ChannelAttention(inp)

    def forward(self, x):
        x_h = self.cov_high(x)
        y = self.avg_pool(x_h)
        y = self.conv_du(y)
        # y = self.ca(x_h)
        x = self.relu(self.conv_1(x))
        x = self.down_c(x_h * y + x)

        return x


class HighFrequencyNode(nn.Module):
    def __init__(self):
        super(HighFrequencyNode, self).__init__()
        # self.theta_phi = HighFrequencyBlock_1(inp=32, oup=32)
        # self.theta_rho = HighFrequencyBlock_1(inp=32, oup=32)
        # self.theta_gam = HighFrequencyBlock_1(inp=32, oup=32)
        # self.theta_eta = HighFrequencyBlock_2(inp=32, oup=32)
        # self.theta_alp = HighFrequencyBlock_2(inp=32, oup=32)
        # self.theta_dta = HighFrequencyBlock_2(inp=32, oup=32)
        # self.theta_phi = Block(dim=32)
        # self.theta_rho = Block(dim=32)
        # self.theta_eta = Block(dim=32)
        self.theta_phi = Block(dim=32)
        self.theta_rho = Block(dim=32)
        self.theta_eta = Block(dim=32)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
        self.sig = nn.Sigmoid()
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        # z2_ = self.theta_phi(z1) + z2
        # z1_ = self.theta_alp(z2) + z1
        # z1 = z1 * torch.exp(self.theta_rho(z2_)) + self.theta_eta(z2_)
        # z2 = z2 * torch.exp(self.theta_gam(z1_)) + self.theta_dta(z1_)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * self.sig(self.theta_rho(z2)) + self.theta_eta(z2)
        # z1 = self.theta_eta(x)
        # z2 = self.theta_rho(z2)

        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        RNNmodules = [HighFrequencyNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*RNNmodules)
        # RNNmodules = [StarNet() for _ in range(num_layers)]
        # self.net = nn.Sequential(*RNNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            # z1_e, z2_e = layer(z1, z2)
            # z1 = z1_e + z1
            # z2 = z2_e + z2
            z1, z2 = layer(z1, z2)
        # z1 = z1_e + z1
        # z2 = z2_e + z2
        return torch.cat((z1, z2), dim=1)
        # for layer in self.net:
        #     x = layer(x)
        # return x

# =============================================================================
import numbers


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# 没有Bias的情况下层正则化
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 方差
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 加上1e-5的目的是防止分母为0
        return x / torch.sqrt(sigma + 1e-5) * self.weight


# 有Bias的情况下层正则化
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        ) # 4+32=36
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True,)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous() # 128 16
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias) #

    def forward(self, hidden_states, inference_params=None):
        with torch.cuda.device(hidden_states.device):
            """
            hidden_states: (B, L, D)
            Returns: same shape as hidden_states
            """
            batch, seqlen, dim = hidden_states.shape # dim is C

            conv_state, ssm_state = None, None
            if inference_params is not None:
                conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
                if inference_params.seqlen_offset > 0:
                    # The states are updated inplace
                    out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                    return out

            # We do matmul and transpose BLH -> HBL at the same time
            xz = rearrange(
                self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj.bias is not None:
                xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

            A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
            # In the backward pass we write dx and dz next to each other to avoid torch.cat
            if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
            else:
                x, z = xz.chunk(2, dim=1)
                # Compute short convolution
                if conv_state is not None:
                    # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x=x,
                        weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        bias=self.conv1d.bias,
                        activation=self.activation,
                    )

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                out = self.out_proj(y)
            return out

    def step(self, hidden_states, conv_state, ssm_state):
        with torch.cuda.device(hidden_states.device):
            dtype = hidden_states.dtype
            assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
            xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
            x, z = xz.chunk(2, dim=-1)  # (B D)

            # Conv step
            if causal_conv1d_update is None:
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
                conv_state[:, :, -1] = x
                x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                if self.conv1d.bias is not None:
                    x = x + self.conv1d.bias
                x = self.act(x).to(dtype=dtype)
            else:
                x = causal_conv1d_update(
                    x,
                    conv_state,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
            dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            # Don't add dt_bias here
            dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
            A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

            # SSM step
            if selective_state_update is None:
                # Discretize A and B
                dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
                dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
                dB = torch.einsum("bd,bn->bdn", dt, B)
                ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
                y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
                y = y + self.D.to(dtype) * x
                y = y * self.act(z)  # (B D)
            else:
                y = selective_state_update(
                    ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
                )

            out = self.out_proj(y)
            return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class CSSM(nn.Module):
    def __init__(self,
                 dim,
                 bias=False,
                 ):
        super(CSSM, self).__init__()
        self.proj = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.dconv = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=bias)
        self.mamba_block = MambaBlock(dim)

    def forward(self, x):
        with torch.cuda.device(x.device):
            x_proj = self.proj(x)
            x_proj = self.dconv(x_proj)
            x_proj = x_proj.flatten(2).transpose(1, 2)
            x_proj = self.mamba_block(x_proj)
            x_proj = x_proj.transpose(1, 2).reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        return x_proj

class Attention2(nn.Module):
    def __init__(self, dim, bias=None):
        super(Attention2, self).__init__()
        self.temperature = nn.Parameter(torch.ones(dim, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.MB = MambaBlock(dim)


    def forward(self, x):
        with torch.cuda.device(x.device):
            b, c, h, w = x.shape

            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            # qk = torch.cat((q,k),dim=1).flatten(2).transpose(1, 2)
            qk = q.flatten(2).transpose(1, 2)
            # qk_filp = torch.flip(torch.cat((q,k),dim=1).flatten(2),dims=[2]).transpose(1, 2)
            qk_filp = torch.flip(q.flatten(2), dims=[2]).transpose(1, 2)
            dep = self.MB(qk).transpose(1, 2).reshape(b, c, h, w)
            dep_filp = torch.flip(self.MB(qk_filp).transpose(1, 2), dims=[2]).reshape(b, c, h, w)
            attn = (dep) * self.temperature
            atten2 = (dep_filp) * self.temperature
            attn = (attn+atten2).softmax(dim=-1)
            # atten2 = atten2.softmax(dim=-1)
            # atten2 = atten2.softmax(dim=-1)
            out = (attn * v)
            # out = ((attn) * v)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w',
        #                 head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
        return out

# class MambaFormer(nn.Module):
#     def __init__(self,
#                  dim,
#                  ffn_expansion_factor,
#                  img_hw=128*128,
#                  bias=False,
#                  LayerNorm_type='WithBias'):
#         super(MambaFormer, self).__init__()
#         self.attn = CSSM(dim)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = ConvolutionalGLU(dim, ffn_expansion_factor)
#     def forward(self, x):
#         with torch.cuda.device(x.device):
#             x = x + self.attn(x)
#             x = x + self.ffn(self.norm2(x))
#
#         return x
class MambaFormer(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 img_hw=128*128,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(MambaFormer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.sssm = SSSM(dim=dim, img_hw=img_hw)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.cssm = CSSM(dim=dim)
        self.ffn =ConvolutionalGLU(dim, ffn_expansion_factor)
        self.dconv_one = nn.Conv2d(dim, dim*2, kernel_size=3, padding=1, bias=bias)
        self.dconv_two = nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        with torch.cuda.device(x.device):
            x_cssm = x + self.cssm(self.norm2(x))
            x_cssm = x_cssm + self.dconv_two(self.dconv_one(x_cssm))
            x_ffn = x_cssm + self.ffn(self.norm1(x_cssm))

        return x_ffn
##########################################################################

class ConvolutionalGLU(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, act_layer=nn.GELU, drop=0.):
        super(ConvolutionalGLU, self).__init__()
        self.fc1 = nn.Conv2d(dim, dim*ffn_expansion_factor, kernel_size=3, stride=1, padding=1, bias=True)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
# Embedding层
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # 64

        self.encoder_level1 = MambaFormer(dim, 2)
        self.encoder_level2 = MambaFormer(dim, 2)
        self.encoder_level4 = MambaFormer(dim, 2)
        self.encoder_level3 = MambaFormer(dim, 2)
        # self.baseFeature = BaseFeatureExtraction(dim=dim, ffn_expansion_factor=2)
        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.baseFeature_s = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)  # 64
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # 64
        out_enc_level2 = self.encoder_level2(out_enc_level1)  # 64
        out_enc_level3 = self.encoder_level3(out_enc_level2)  # 64
        out_enc_level4 = self.encoder_level4(out_enc_level3)  # 64
        base_feature = self.baseFeature(out_enc_level4)
        base_feature = self.baseFeature_s(base_feature)
        detail_feature = self.detailFeature(out_enc_level4)
        return base_feature, detail_feature, out_enc_level4  # feature_V or feature_D


class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 isPhaseI=True):

        super(Restormer_Decoder, self).__init__()

        self.reduce_channel = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = MambaFormer(dim, 2)
        self.encoder_level3 = MambaFormer(dim, 2)
        self.encoder_level4 = MambaFormer(dim, 2)
        self.encoder_level5 = MambaFormer(dim, 2)

        # Transformer最后的全连接层
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        # 激活函数
        self.sigmoid = nn.Sigmoid()

    # 多传入了可见光图像
    #                          low-frequency    high-frequency
    def forward(self, inp_img, base_feature, detail_feature, feature_vis, feature_ir):  # 8 128 128 128
        if feature_vis is not None and feature_ir is not None:
            feature_combine = feature_vis + feature_ir  # 64
            out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)  # 128
            out_enc_level0 = self.reduce_channel(out_enc_level0) + feature_combine  # 64 + 64
            # 进入Transformer                                 64              64
            out_enc_level1 = self.encoder_level2(out_enc_level0)
            out_enc_level2 = self.encoder_level3(out_enc_level1)
            out_enc_level3 = self.encoder_level4(out_enc_level2)
            out_enc_level4 = self.encoder_level5(out_enc_level3)
            out_enc_level4 = self.output(out_enc_level4) + inp_img
        else:
            out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
            out_enc_level0 = self.reduce_channel(out_enc_level0)  # 64
            out_enc_level1 = self.encoder_level2(out_enc_level0)
            out_enc_level2 = self.encoder_level3(out_enc_level1)
            out_enc_level3 = self.encoder_level4(out_enc_level2)
            out_enc_level4 = self.encoder_level5(out_enc_level3)
            out_enc_level4 = self.output(out_enc_level4) + inp_img
        return self.sigmoid(out_enc_level4), out_enc_level4


class feature_measurement(nn.Module):
    def __init__(self):
        super(feature_measurement, self).__init__()

    def features_grad(self, features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.cuda()
        _, c, _, _ = features.shape
        c = int(c)
        for i in range(c):
            feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
            if i == 0:
                feat_grads = feat_grad
            else:
                feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
        return feat_grads

    def forward(self, base_feature, detail_feature):
        m1 = torch.mean(self.features_grad(base_feature).pow(2), dim=[1, 2, 3])
        m2 = torch.mean(self.features_grad(detail_feature).pow(2), dim=[1, 2, 3])

        w1 = torch.unsqueeze(m1, dim=-1)  # (batch_size,1)
        w2 = torch.unsqueeze(m2, dim=-1)  # (batch_size,1)

        c = 3500
        weight_1 = torch.mean(w1, dim=0) / c
        weight_2 = torch.mean(w2, dim=0) / c
        weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
        weight_list = F.softmax(weight_list, dim=-1)
        return weight_list


class feature_weight_map(nn.Module):
    def __init__(self,
                 int_channel=1,
                 out_channel=1,
                 dim=64,
                 bias=False,
                 LayerNorm_type='WithBias',
                 drop_rate=0.2,
                 K=32,
                 num_head=8):
        super(feature_weight_map, self).__init__()
        self.attn = Attention(dim * 2, num_head, bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type)

        self.norm1 = LayerNorm(int_channel, LayerNorm_type)  # 1
        self.conv1 = nn.Conv2d(int_channel, dim, stride=1, kernel_size=3, padding=1, bias=bias)  # 64

        self.norm2 = LayerNorm(dim, LayerNorm_type)  # 64
        self.conv2 = nn.Conv2d(dim, K, stride=1, padding=1, kernel_size=3, bias=bias)

        self.norm3 = LayerNorm(dim + K, LayerNorm_type)
        self.conv3 = nn.Conv2d(dim + K, K, stride=1, padding=1, kernel_size=3, bias=bias)

        self.norm4 = LayerNorm(dim + 2 * K, LayerNorm_type)
        self.conv4 = nn.Conv2d(dim + 2 * K, K, stride=1, padding=1, kernel_size=3, bias=bias)

        # Transition
        self.norm5 = LayerNorm(dim + 3 * K, LayerNorm_type)
        self.conv5 = nn.Conv2d(dim + 3 * K, dim * 2, kernel_size=1, stride=1, bias=False)
        # self.drop_rate = drop_rate

        self.conv6 = nn.Conv2d(dim * 2, dim * 2, stride=1, padding=1, kernel_size=3, bias=bias)
        self.norm6 = LayerNorm(dim * 2, LayerNorm_type)
        self.conv7 = nn.Conv2d(dim * 2, dim, stride=1, padding=1, kernel_size=3, bias=bias)
        self.norm7 = LayerNorm(dim, LayerNorm_type)

    def lrelu(self, x, leak=0.2):
        return torch.max(x, leak * x)

    def dense_concat(self, x, *args):
        for i in args:
            x = torch.cat((x, i), dim=1)
        return x

    def forward(self, img_vi, img_ir):
        feature_vi_1 = self.conv1(self.lrelu(self.norm1(img_vi)))  # 64
        feature_vi_2 = self.conv2(self.lrelu(self.norm2(feature_vi_1)))  # 32
        feature_vi_3 = self.conv3(self.lrelu(self.norm3(self.dense_concat(feature_vi_2, feature_vi_1))))  # 32
        feature_vi_4 = self.conv4(
            self.lrelu(self.norm4(self.dense_concat(feature_vi_3, feature_vi_1, feature_vi_2))))  # 32
        feature_vi_5 = self.conv5(
            self.lrelu(self.norm5(self.dense_concat(feature_vi_4, feature_vi_3, feature_vi_2, feature_vi_1))))  # 32

        feature_ir_1 = self.conv1(self.lrelu(self.norm1(img_ir)))
        feature_ir_2 = self.conv2(self.lrelu(self.norm2(feature_ir_1)))
        feature_ir_3 = self.conv3(self.lrelu(self.norm3(self.dense_concat(feature_ir_2, feature_ir_1))))
        feature_ir_4 = self.conv4(self.lrelu(self.norm4(self.dense_concat(feature_ir_3, feature_ir_2, feature_ir_1))))
        feature_ir_5 = self.conv5(
            self.lrelu(self.norm5(self.dense_concat(feature_ir_4, feature_ir_3, feature_ir_2, feature_ir_1))))
        feature_ir_5 = self.attn(feature_ir_5)
        feature_ir_5 = self.norm(feature_ir_5)

        feature_vi = self.lrelu(self.norm6(self.conv6(feature_vi_5)))
        feature_vi = self.norm7(self.conv7(feature_vi))
        feature_ir = self.lrelu(self.norm6(self.conv6(feature_ir_5)))
        feature_ir = self.norm7(self.conv7(feature_ir))

        # vector_feature_vi_weight = F.softmax(torch.mean(feature_vi, dim=[2, 3], keepdim=True), dim=1)
        # vector_feature_ir_weight = F.softmax(torch.mean(feature_ir, dim=[2, 3], keepdim=True), dim=1)
        # vector_feature_vi_weight = F.softmax(feature_vi, dim=-1)
        # vector_feature_ir_weight = F.softmax(feature_ir, dim=-1)
        # feature_vi = torch.multiply(vector_feature_vi_weight, feature_vi)
        # feature_ir = torch.multiply(vector_feature_ir_weight, feature_ir)

        return feature_vi, feature_ir  # 8 64 128 128


def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)


def LoG(img):
    window_size = 9
    window = torch.Tensor([[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                            [1, 2, 4, 5, 5, 5, 4, 2, 1],
                            [1, 4, 5, 3, 0, 3, 5, 4, 1],
                            [2, 5, 3, -12, -24, -12, 3, 5, 2],
                            [2, 5, 0, -24, -40, -24, 0, 5, 2],
                            [2, 5, 3, -12, -24, -12, 3, 5, 2],
                            [1, 4, 5, 3, 0, 3, 4, 4, 1],
                            [1, 2, 4, 5, 5, 5, 4, 2, 1],
                            [0, 1, 1, 2, 2, 2, 1, 1, 0]]]).cuda()
    channel = img.shape[1]
    window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    output = F.conv2d(img, window, padding=window_size // 2, groups=channel)
    output = minmaxscaler(output)  # 归一化到0~1之间
    return output  # Log(X(x,y))

class feature_enhancement(nn.Module):
    def __init__(self,
                 dim=64,
                 int_channel=1,
                 out_channel=2,
                 bias=False):
        super(feature_enhancement, self).__init__()
        self.dim = dim
        self.relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(int_channel, dim, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=bias)
        self.conv6 = nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=bias)
        self.conv7 = nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=bias)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=dim)

    def low_frequency_decomposition(self, x):
        return x - LoG(x)

    def high_enhancement(self, x):
        return x + LoG(x)

    def low_enhancement(self, x):
        return x + self.low_frequency_decomposition(x)

    def forward(self, img_V, img_I):  # 8 1 128 128
        # first block
        low_enhancement = self.low_enhancement(img_I)
        low_enhancement = self.relu(self.norm(self.conv1(low_enhancement)))
        inlow_low_frequency_feature = self.low_frequency_decomposition(low_enhancement)  # 64
        inlow_high_frequency_feature = LoG(low_enhancement)  # 64
        high_enhancement = self.high_enhancement(img_V)
        high_enhancement = self.relu(self.norm(self.conv1(high_enhancement)))
        inhigh_low_frequency_feature = self.low_frequency_decomposition(high_enhancement)  # 64
        inhigh_high_frequency_feature = LoG(high_enhancement)  # 64

        # second block
        low_frequency_feature = torch.cat((inlow_low_frequency_feature, inhigh_low_frequency_feature), dim=1)  # 128
        high_frequency_feature = torch.cat((inlow_high_frequency_feature, inhigh_high_frequency_feature), dim=1)  # 128
        low_enhancement_s = self.low_enhancement(low_frequency_feature)
        low_enhancement_s = self.relu(self.norm(self.conv2(low_enhancement_s)))
        inlow_low_frequency_feature_s = self.low_frequency_decomposition(low_enhancement_s)  # 64
        inlow_high_frequency_feature_s = LoG(low_enhancement_s)  # 64
        high_enhancement_s = self.high_enhancement(high_frequency_feature)
        high_enhancement_s = self.relu(self.norm(self.conv3(high_enhancement_s)))
        inhigh_low_frequency_feature_s = self.low_frequency_decomposition(high_enhancement_s)  # 64
        inhigh_high_frequency_feature_s = LoG(high_enhancement_s)  # 64

        # third block
        low_frequency_feature = torch.cat(
            (inlow_low_frequency_feature, inlow_low_frequency_feature_s, inhigh_low_frequency_feature_s), dim=1)  # 192
        high_frequency_feature = torch.cat(
            (inhigh_high_frequency_feature, inlow_high_frequency_feature_s, inhigh_high_frequency_feature_s),
            dim=1)  # 192
        low_enhancement_t = self.low_enhancement(low_frequency_feature)
        low_enhancement_t = self.relu(self.norm(self.conv4(low_enhancement_t)))  # 64
        low_frequency_feature = low_enhancement_t
        high_enhancement_t = self.high_enhancement(high_frequency_feature)
        high_enhancement_t = self.relu(self.norm(self.conv5(high_enhancement_t)))  # 64
        high_frequency_feature = high_enhancement_t

        return low_frequency_feature, high_frequency_feature  #


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return 1 - self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return 1 - self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 3, 1, 1, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 3, 1, 1, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + x
        return x

class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=2.,
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp2(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        # 将输入张量（input）沿着指定维度（dim）均匀的分割成特定数量的张量块（chunks），并返回元素为张量块的元组。
        q, k, v = qkv.chunk(3, dim=1)
        # rearrange用于对张量的维度进行重新变换排序
        #                   head × c  ->             h × w
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        # 正则化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # 点积注意力？
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 矩阵乘法
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp2(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        # 隐藏层特征维度等于输入维度乘以扩张因子
        hidden_features = int(in_features * ffn_expansion_factor)
        # 1*1 升维
        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)
        # 3*3 分组卷积
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)
        # 1*1 降维
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


import torch
import torch.nn as nn

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(dim, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        # 将输入张量（input）沿着指定维度（dim）均匀的分割成特定数量的张量块（chunks），并返回元素为张量块的元组。
        q, k, v = qkv.chunk(3, dim=1)
        # rearrange用于对张量的维度进行重新变换排序
        #                   head × c  ->             h × w
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)',
        #               head=self.num_heads)
        # 正则化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # 点积注意力？
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # 矩阵乘法
        out = (attn @ v)
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w',
        #                 head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out

# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# 卷积 Transformer 模块
class Vit(nn.Module):
    def __init__(self, in_channels, dim, num_heads=1):
        super(Vit, self).__init__()
        self.conv_block = ConvBlock(in_channels, dim)
        self.attention = MultiHeadAttention(dim)

    def forward(self, x):
        with torch.cuda.device(x.device):
            x = self.attention(x)
            # 卷积操作
            x = self.conv_block(x)
            # B, C, H, W = x.shape
            # 将特征图展平为序列
            # x = x.flatten(2).transpose(1, 2)
            # 多头自注意力操作

            # 将序列还原为特征图
            # x = x.transpose(1, 2).reshape(B, C, H, W)
            return x
