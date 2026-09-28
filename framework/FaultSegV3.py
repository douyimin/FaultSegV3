"""hrnet_highres 的推理内存优化版本。

与 hrnet_highres.py 逐语义一致 (通道数 / 模块结构 / state_dict 键 / DropPath / Sigmoid 输出
/ 全网只有 skip_layer2 一次下采样), 额外提供仅用于推理的分块 (chunked) 前向路径:

    model = FaultSegV3(c=32).eval()
    y0 = model(x, rank=0)                 # 与原实现完全一致的前向
    y1 = model(x, rank=1, chunk_size=64)  # 分块高风险算子 (上采样 / 反卷积)
    y2 = model(x, rank=2, chunk_size=64)  # 额外融合并分块 dec_out
    y3 = model(x, rank=3, chunk_size=64)  # 额外丢弃并按需重算全分辨率 skip_x1
    y4 = model(x, rank=4, chunk_size=64)  # 最省内存: 不落地 skip_x1, 也不落地全分辨率 dec_block1 输出

训练模式默认使用原始前向，推理模式默认使用 rank=4。
rank>0 仅用于推理 (eval + no_grad)。
"""

import itertools
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

Int3 = Tuple[int, int, int]


def _to_3tuple(x: Union[int, Sequence[int]]) -> Int3:
    if isinstance(x, int):
        return (x, x, x)
    x = tuple(int(v) for v in x)
    assert len(x) == 3
    return x


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _conv_out_size_1d(in_size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _tile_index_3d(i: int, j: int, k: int, block_size: int, shape: Int3, halo: int):
    """返回 (有效区域, 带 halo 的取块区域, 块内裁剪区域, F.pad 列表)。

    5D 张量的 F.pad 顺序为 [w0, w1, h0, h1, d0, d1]。
    """
    d, h, w = shape
    vd0, vh0, vw0 = i * block_size, j * block_size, k * block_size
    vd1, vh1, vw1 = min(vd0 + block_size, d), min(vh0 + block_size, h), min(vw0 + block_size, w)

    pd0, ph0, pw0 = max(0, vd0 - halo), max(0, vh0 - halo), max(0, vw0 - halo)
    pd1, ph1, pw1 = min(d, vd1 + halo), min(h, vh1 + halo), min(w, vw1 + halo)

    pad_d0 = max(0, halo - (vd0 - pd0))
    pad_h0 = max(0, halo - (vh0 - ph0))
    pad_w0 = max(0, halo - (vw0 - pw0))
    pad_d1 = max(0, halo - (pd1 - vd1))
    pad_h1 = max(0, halo - (ph1 - vh1))
    pad_w1 = max(0, halo - (pw1 - vw1))
    padlist = [pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1]

    rd0 = vd0 - pd0 + pad_d0
    rh0 = vh0 - ph0 + pad_h0
    rw0 = vw0 - pw0 + pad_w0
    rd1 = rd0 + (vd1 - vd0)
    rh1 = rh0 + (vh1 - vh0)
    rw1 = rw0 + (vw1 - vw0)

    return (vd0, vh0, vw0, vd1, vh1, vw1), (pd0, ph0, pw0, pd1, ph1, pw1), (rd0, rh0, rw0, rd1, rh1, rw1), padlist


def set_pad_as_zero(x: Tensor, padlist: Sequence[int]) -> Tensor:
    """把人为 padding 出来的 halo 区域重新置零。

    当一个融合块内含多次带 padding 的卷积时必须这样做: 否则第一次卷积会在真实体边界之外
    产生非零响应, 下一次卷积会错误地把它当成真实数据用掉。
    """
    if padlist[0] > 0:
        x[:, :, :, :, :padlist[0]] = 0
    if padlist[1] > 0:
        x[:, :, :, :, -padlist[1]:] = 0
    if padlist[2] > 0:
        x[:, :, :, :padlist[2], :] = 0
    if padlist[3] > 0:
        x[:, :, :, -padlist[3]:, :] = 0
    if padlist[4] > 0:
        x[:, :, :padlist[4], :, :] = 0
    if padlist[5] > 0:
        x[:, :, -padlist[5]:, :, :] = 0
    return x


@torch.no_grad()
def interpolate3d_chunked(
        x: Tensor,
        block_size: int = 64,
        scale_factor: int = 2,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        size: Optional[Int3] = None,
) -> Tensor:
    """算子级分块插值。

    这与 patch 级推理不同: 只有插值算子本身被分块, 上下游张量仍是整体张量。
    """
    assert x.ndim == 5, "只支持 5D 张量 (N, C, D, H, W)。"
    if mode == "nearest":
        align_corners = None
        halo = 0
    else:
        halo = 1  # 三线性插值局部只需 1 个体素的 halo

    b, c, d, h, w = x.shape
    if size is None:
        out_d, out_h, out_w = d * scale_factor, h * scale_factor, w * scale_factor
    else:
        out_d, out_h, out_w = size
        if (out_d, out_h, out_w) != (d * scale_factor, h * scale_factor, w * scale_factor):
            # 非整数或非均匀缩放在本网络中不会出现, 保险起见回退到 PyTorch 实现。
            return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    out = torch.empty((b, c, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    tile = max(1, block_size - 2 * halo)
    nd, nh, nw = _ceil_div(d, tile), _ceil_div(h, tile), _ceil_div(w, tile)

    s = scale_factor
    for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
        (vd0, vh0, vw0, vd1, vh1, vw1), (pd0, ph0, pw0, pd1, ph1, pw1), (rd0, rh0, rw0, rd1, rh1, rw1), _ = \
            _tile_index_3d(ii, jj, kk, tile, (d, h, w), halo)
        patch = x[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
        patch = F.interpolate(patch, scale_factor=s, mode=mode, align_corners=align_corners)
        out[:, :, vd0 * s:vd1 * s, vh0 * s:vh1 * s, vw0 * s:vw1 * s] = \
            patch[:, :, rd0 * s:rd1 * s, rh0 * s:rh1 * s, rw0 * s:rw1 * s]
    return out


class NearestUpsample3d(nn.Module):
    """nn.Upsample(mode='nearest') 的等价替换, 支持可选的分块执行。

    无参数模块, 因此不影响 state_dict 的键与形状。
    """

    def __init__(self, scale_factor: float = 2.0, chunk_size: Optional[int] = None):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if self.chunk_size is None:
            return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return interpolate3d_chunked(x, self.chunk_size, self.scale_factor, mode="nearest")

    def extra_repr(self) -> str:
        return f'scale_factor={self.scale_factor}, chunk_size={self.chunk_size}'


@torch.no_grad()
def conv3d_chunked(x: Tensor, conv: nn.Conv3d, block_size: int = 128) -> Tensor:
    """带正确 halo padding 的 Conv3d 分块执行, 数值上等价于 conv(x)。

    仅用于推理, 假定 dilation=1 且下采样倍率为整数。
    """
    assert not conv.training, "分块 Conv3d 仅用于 eval/推理。"
    assert x.ndim == 5

    b, _, d, h, w = x.shape
    kernel = _to_3tuple(conv.kernel_size)
    stride = _to_3tuple(conv.stride)
    padding = _to_3tuple(conv.padding)
    dilation = _to_3tuple(conv.dilation)
    assert dilation == (1, 1, 1), "该辅助函数目前假定 dilation=1。"

    out_d = _conv_out_size_1d(d, kernel[0], stride[0], padding[0], dilation[0])
    out_h = _conv_out_size_1d(h, kernel[1], stride[1], padding[1], dilation[1])
    out_w = _conv_out_size_1d(w, kernel[2], stride[2], padding[2], dilation[2])

    scale_d, scale_h, scale_w = d // out_d, h // out_h, w // out_w
    assert (d % out_d, h % out_h, w % out_w) == (0, 0, 0)

    old_padding = conv.padding
    old_padding_mode = conv.padding_mode
    conv.padding = (0, 0, 0)
    if old_padding_mode == "zeros":
        pad_mode, pad_value = "constant", 0.0
    else:
        pad_mode, pad_value = old_padding_mode, None
        conv.padding_mode = "zeros"

    out = torch.empty((b, conv.out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    tile = max(stride[0], block_size)
    nd, nh, nw = _ceil_div(d, tile), _ceil_div(h, tile), _ceil_div(w, tile)

    try:
        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            vd0, vh0, vw0 = ii * tile, jj * tile, kk * tile
            vd1, vh1, vw1 = min(vd0 + tile, d), min(vh0 + tile, h), min(vw0 + tile, w)

            pd0, ph0, pw0 = max(0, vd0 - padding[0]), max(0, vh0 - padding[1]), max(0, vw0 - padding[2])
            pd1, ph1, pw1 = min(d, vd1 + padding[0]), min(h, vh1 + padding[1]), min(w, vw1 + padding[2])

            padlist = [0, 0, 0, 0, 0, 0]
            if vd0 == pd0:
                padlist[4] = padding[0]
            if vh0 == ph0:
                padlist[2] = padding[1]
            if vw0 == pw0:
                padlist[0] = padding[2]
            if vd1 == pd1:
                padlist[5] = padding[0]
            if vh1 == ph1:
                padlist[3] = padding[1]
            if vw1 == pw1:
                padlist[1] = padding[2]

            patch = x[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
            if sum(padlist) > 0:
                patch = F.pad(patch, padlist, mode=pad_mode, value=pad_value)
            patch = conv(patch)

            od0, oh0, ow0 = vd0 // scale_d, vh0 // scale_h, vw0 // scale_w
            od1, oh1, ow1 = vd1 // scale_d, vh1 // scale_h, vw1 // scale_w
            out[:, :, od0:od1, oh0:oh1, ow0:ow1] = patch
    finally:
        conv.padding = old_padding
        conv.padding_mode = old_padding_mode
    return out


@torch.no_grad()
def conv_transpose3d_chunked(x: Tensor, deconv: nn.ConvTranspose3d, block_size: int = 64) -> Tensor:
    """解码器中 ConvTranspose3d 的分块执行。"""
    assert not deconv.training, "分块 ConvTranspose3d 仅用于 eval/推理。"
    assert x.ndim == 5

    b, _, d, h, w = x.shape
    stride = _to_3tuple(deconv.stride)
    padding = _to_3tuple(deconv.padding)
    kernel = _to_3tuple(deconv.kernel_size)
    dilation = _to_3tuple(deconv.dilation)
    output_padding = _to_3tuple(deconv.output_padding)

    out_d = (d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + output_padding[0] + 1
    out_h = (h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + output_padding[1] + 1
    out_w = (w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel[2] - 1) + output_padding[2] + 1

    scale_d, scale_h, scale_w = out_d // d, out_h // h, out_w // w
    assert (out_d % d, out_h % h, out_w % w) == (0, 0, 0)

    halo = max(padding)
    tile = max(1, block_size - 2 * halo)
    out = torch.empty((b, deconv.out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    nd, nh, nw = _ceil_div(d, tile), _ceil_div(h, tile), _ceil_div(w, tile)

    for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
        vd0, vh0, vw0 = ii * tile, jj * tile, kk * tile
        vd1, vh1, vw1 = min(vd0 + tile, d), min(vh0 + tile, h), min(vw0 + tile, w)

        pd0, ph0, pw0 = max(0, vd0 - halo), max(0, vh0 - halo), max(0, vw0 - halo)
        pd1, ph1, pw1 = min(d, vd1 + halo), min(h, vh1 + halo), min(w, vw1 + halo)

        rd0, rh0, rw0 = (vd0 - pd0) * scale_d, (vh0 - ph0) * scale_h, (vw0 - pw0) * scale_w
        rd1 = rd0 + (vd1 - vd0) * scale_d
        rh1 = rh0 + (vh1 - vh0) * scale_h
        rw1 = rw0 + (vw1 - vw0) * scale_w

        od0, oh0, ow0 = vd0 * scale_d, vh0 * scale_h, vw0 * scale_w
        od1, oh1, ow1 = vd1 * scale_d, vh1 * scale_h, vw1 * scale_w

        patch = deconv(x[:, :, pd0:pd1, ph0:ph1, pw0:pw1])
        out[:, :, od0:od1, oh0:oh1, ow0:ow1] = patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]
    return out


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """对每个样本独立地随机丢弃整条残差分支 (Stochastic Depth)。

    训练时按概率 drop_prob 将整条分支输出置零, 并用 1/(1-drop_prob) 缩放保持期望;
    推理时不做任何修改。
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath / Stochastic Depth 的模块封装。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path_prob=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm3d(planes * self.expansion)
        self.act_fun = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act_fun(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 只对残差分支应用 DropPath, 主路径 (shortcut) 始终保留
        out = self.drop_path(out)
        out += residual
        out = self.act_fun(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_path_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(planes)
        self.act_fun = nn.SiLU(inplace=True)
        # 与 hrnet_highres 保持一致: 第二个卷积的 in_channels 也是 inplanes
        # (该网络中所有 BasicBlock 都满足 inplanes == planes, 因此两者等价)
        self.conv2 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 只对残差分支应用 DropPath, 主路径 (shortcut) 始终保留
        out = self.drop_path(out)
        out += residual
        out = self.act_fun(out)

        return out


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, drop_path_probs=None, upsample_chunk_size: Optional[int] = None):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.upsample_chunk_size = upsample_chunk_size

        # 每个 StageModule 内部每条 branch 有 4 个 BasicBlock, 共享同一组 drop_path 概率
        if drop_path_probs is None:
            drop_path_probs = [0.0, 0.0, 0.0, 0.0]
        assert len(drop_path_probs) == 4, "StageModule 每条分支固定 4 个 BasicBlock"

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, drop_path_prob=drop_path_probs[0]),
                BasicBlock(w, w, drop_path_prob=drop_path_probs[1]),
                BasicBlock(w, w, drop_path_prob=drop_path_probs[2]),
                BasicBlock(w, w, drop_path_prob=drop_path_probs[3]),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # 占位, 因为 Sequential 是可调用的
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
                        nn.BatchNorm3d(c * (2 ** i)),
                        NearestUpsample3d(scale_factor=(2.0 ** (j - i)), chunk_size=upsample_chunk_size),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                      padding=(1, 1, 1), bias=False),
                            nn.BatchNorm3d(c * (2 ** j)),
                            nn.SiLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                  padding=(1, 1, 1), bias=False),
                        nn.BatchNorm3d(c * (2 ** i)),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.act_fun = nn.SiLU(inplace=True)

    def set_upsample_chunk_size(self, chunk_size: Optional[int]):
        self.upsample_chunk_size = chunk_size
        for m in self.modules():
            if isinstance(m, NearestUpsample3d):
                m.chunk_size = chunk_size

    def forward(self, x):
        assert len(self.branches) == len(x)

        # 就地替换 branch 输出, 避免同时持有输入和输出两份列表
        x = list(x)
        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fused = []
        for i in range(len(self.fuse_layers)):
            y = None
            y_is_alias = False  # y 是否直接引用了 x[j] (i == j 时 fuse 层是恒等映射)
            target_size = None
            for j in range(len(self.branches)):
                z = self.fuse_layers[i][j](x[j])
                if y is None:
                    y = z
                    y_is_alias = z is x[j]
                    target_size = z.shape[2:]
                else:
                    if z.shape[2:] != target_size:
                        # 奇数尺寸时 nearest 上采样可能与目标差 1, 这里对齐回目标尺寸
                        if self.upsample_chunk_size is None:
                            z = F.interpolate(z, size=target_size, mode="nearest")
                        else:
                            z = interpolate3d_chunked(
                                z, block_size=self.upsample_chunk_size, mode="nearest", size=target_size,
                            )
                    # 原实现为 y = y + z; 推理时若 y 不是 x[j] 的别名, 可就地累加省一份临时张量
                    if y_is_alias or torch.is_grad_enabled():
                        y = y + z
                        y_is_alias = False
                    else:
                        y = y.add_(z)
                    del z
            if y_is_alias:
                # 不能对仍被其它输出分支使用的 x[j] 做 inplace 激活
                y = F.silu(y)
            else:
                y = self.act_fun(y)
            x_fused.append(y)

        # 融合完成后立刻释放 branch 输出
        for i in range(len(x)):
            x[i] = None
        return x_fused


class FaultSegV3(nn.Module):
    """高分辨率 3D FaultSegV3, 附带仅用于推理的低内存路径。

    结构与原 FaultSegV3 完全一致（state_dict 可直接加载）：
    全网只有 skip_layer2 一次下采样, 主干最高分辨率为 1/2, 解码器只有一次上采样。
    """

    def __init__(self, c=8, drop_path_rate=0.1, out_channels: int = 1):
        super(FaultSegV3, self).__init__()
        self.base = c
        self.out_channels = out_channels

        # ===== DropPath 概率线性增加 (从 layer1 浅层到 stage3 深层) =====
        # 总共 28 个带残差的 block:
        #   layer1: 4 个 Bottleneck
        #   stage2: 2 个 StageModule × 4 个 BasicBlock = 8
        #   stage3: 4 个 StageModule × 4 个 BasicBlock = 16
        total_blocks = 4 + 2 * 4 + 4 * 4  # = 28
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, total_blocks)]
        dpr_layer1 = dpr[0:4]
        dpr_stage2 = dpr[4:4 + 8]
        dpr_stage3 = dpr[4 + 8:]

        # ===== UNet 风格的 skip 分支 (替代原 stem 的 conv1) =====
        # skip_layer1: 全分辨率, 1 -> c 通道
        self.skip_layer1 = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
        )
        # skip_layer2: 1/2 分辨率, c -> c*2 通道 (全网唯一的一次下采样)
        self.skip_layer2 = nn.Sequential(
            nn.Conv3d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c * 2),
            nn.SiLU(inplace=True),
            nn.Conv3d(c * 2, c * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c * 2),
            nn.SiLU(inplace=True),
        )

        # Input (stem net) - 只在 skip_layer2 里下采样一次, 这里 conv2 保持 1/2 分辨率 (stride=1)
        self.conv2 = nn.Conv3d(c * 2, c * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        self.norm2 = nn.BatchNorm3d(c * 2)
        self.act_fun = nn.SiLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv3d(c * 2, c * 8, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(c * 8),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(c * 2, c * 2, downsample=downsample, drop_path_prob=dpr_layer1[0]),
            Bottleneck(c * 8, c * 2, drop_path_prob=dpr_layer1[1]),
            Bottleneck(c * 8, c * 2, drop_path_prob=dpr_layer1[2]),
            Bottleneck(c * 8, c * 2, drop_path_prob=dpr_layer1[3]),
        )

        # Fusion layer 1 (transition1)      - 建立前两条分支
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(c * 8, c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(c),
                nn.SiLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # 双层 Sequential 以匹配官方预训练权重
                nn.Conv3d(c * 8, c * (2 ** 1), kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(c * (2 ** 1)),
                nn.SiLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, drop_path_probs=dpr_stage2[0:4]),
            StageModule(stage=2, output_branches=2, c=c, drop_path_probs=dpr_stage2[4:8]),
            StageModule(stage=2, output_branches=2, c=c, drop_path_probs=dpr_stage3[0:4]),
            StageModule(stage=2, output_branches=2, c=c, drop_path_probs=dpr_stage3[4:8]),
            StageModule(stage=2, output_branches=2, c=c, drop_path_probs=dpr_stage3[8:12]),
            StageModule(stage=2, output_branches=2, c=c, drop_path_probs=dpr_stage3[12:16]),
        )


        # ----- 解码器 -----
        # 只下采样一次, 所以主干最高分辨率分支已是 1/2, 解码器只需一次上采样
        # 主干输出 cat 后通道数 = c + c*2 = 3c (1/2 分辨率), 再 cat skip_x2 (c*2 ch, 1/2) -> 5c
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose3d(c * 3 + c * 2, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
        )

        self.dec_out = nn.Sequential(
            nn.Conv3d(c + c, c, kernel_size=3, padding=1),  # cat skip_x1 后通道 = c + c
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.SiLU(inplace=True),
            nn.Conv3d(c, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def set_upsample_chunk_size(self, chunk_size: Optional[int]):
        for m in self.modules():
            if isinstance(m, StageModule):
                m.set_upsample_chunk_size(chunk_size)
            elif isinstance(m, NearestUpsample3d):
                m.chunk_size = chunk_size


    def _encode_backbone(self, skip_x2: Tensor, chunk_size: Optional[int] = None) -> Tensor:
        """主干 (conv2 -> layer1 -> stage2 -> 分支拼接), 输出 1/2 分辨率 3c 通道。"""
        x = self.conv2(skip_x2)
        x = self.norm2(x)
        x = self.act_fun(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)
        if chunk_size is None:
            up = F.interpolate(x[1], size=x[0].shape[2:], mode="nearest")
        else:
            up = interpolate3d_chunked(x[1], chunk_size, 2, mode="nearest", size=x[0].shape[2:])
        x[1] = None
        return torch.cat([x[0], up], dim=1)  # 1/2 分辨率, 3c 通道

    def forward_raw(self, x: Tensor) -> Tensor:
        """与原 FaultSegV3.forward 逐语义一致的原始前向。"""
        # ===== UNet 风格 skip 分支 (替代原 stem 的 conv1, 同时保存供解码器使用) =====
        skip_x1 = self.skip_layer1(x)  # 全分辨率, c 通道
        skip_x2 = self.skip_layer2(skip_x1)  # 1/2 分辨率, c*2 通道 (唯一一次下采样)

        # ===== 主干: 从 skip_x2 接入 conv2 (保持 1/2 分辨率) =====
        x = self._encode_backbone(skip_x2, chunk_size=None)

        # ===== 解码器: 1/2 分辨率先 cat skip_x2, 再一次上采样到全分辨率 =====
        x = torch.cat([x, skip_x2], dim=1)
        x = self.dec_block1(x)  # -> 全分辨率
        x = torch.cat([x, skip_x1], dim=1)
        return self.dec_out(x)

    @torch.no_grad()
    def _skip1_region(self, inp: Tensor, patch: Tuple[int, int, int, int, int, int], halo: int = 2) -> Tensor:
        """只为指定的全分辨率区域重算 skip_layer1。"""
        _, _, d, h, w = inp.shape
        pd0, ph0, pw0, pd1, ph1, pw1 = patch

        qd0, qh0, qw0 = max(0, pd0 - halo), max(0, ph0 - halo), max(0, pw0 - halo)
        qd1, qh1, qw1 = min(d, pd1 + halo), min(h, ph1 + halo), min(w, pw1 + halo)
        pad_d0 = max(0, halo - (pd0 - qd0))
        pad_h0 = max(0, halo - (ph0 - qh0))
        pad_w0 = max(0, halo - (pw0 - qw0))
        pad_d1 = max(0, halo - (qd1 - pd1))
        pad_h1 = max(0, halo - (qh1 - ph1))
        pad_w1 = max(0, halo - (qw1 - pw1))
        padlist = [pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1]

        q = inp[:, :, qd0:qd1, qh0:qh1, qw0:qw1]
        if sum(padlist) > 0:
            q = F.pad(q, padlist, mode="constant", value=0.0)

        # 手动执行 skip_layer1, 每个局部卷积块后把人为 halo 重新置零
        y = self.skip_layer1[0](q)
        y = self.skip_layer1[1](y)
        y = self.skip_layer1[2](y)
        y = set_pad_as_zero(y, padlist)
        y = self.skip_layer1[3](y)
        y = self.skip_layer1[4](y)
        y = self.skip_layer1[5](y)
        y = set_pad_as_zero(y, padlist)

        rd0 = pd0 - qd0 + pad_d0
        rh0 = ph0 - qh0 + pad_h0
        rw0 = pw0 - qw0 + pad_w0
        rd1 = rd0 + (pd1 - pd0)
        rh1 = rh0 + (ph1 - ph0)
        rw1 = rw0 + (pw1 - pw0)
        return y[:, :, rd0:rd1, rh0:rh1, rw0:rw1]

    @torch.no_grad()
    def _skip2_from_input_chunked(self, inp: Tensor, chunk_size: int) -> Tensor:
        """在不落地全分辨率 skip_x1 的前提下计算 skip_layer2。

        结果与 ``self.skip_layer2(self.skip_layer1(inp))`` 在浮点误差内一致: 按 1/2 分辨率
        输出网格分块, 每块重算所需的全分辨率 skip_layer1 上下文, 并裁掉受人为块边界影响的区域。
        """
        assert not self.training, "_skip2_from_input_chunked 只在 eval 模式下有效。"
        b, _, d, h, w = inp.shape
        # skip_layer2[0] 是 stride=2, kernel=3, padding=1 的卷积, 输出尺寸是 ceil(in/2)
        d2 = _conv_out_size_1d(d, 3, 2, 1, 1)
        h2 = _conv_out_size_1d(h, 3, 2, 1, 1)
        w2 = _conv_out_size_1d(w, 3, 2, 1, 1)
        out_channels = self.skip_layer2[-2].num_features  # 等于 c*2, 写成这样以便 stem 改动时仍然正确
        out = torch.empty((b, out_channels, d2, h2, w2), device=inp.device, dtype=inp.dtype)

        # 1/2 分辨率网格上的保守 halo, 覆盖 skip_layer2 的两次卷积以及 skip_layer1 的两次卷积
        half_halo = 6
        nd, nh, nw = _ceil_div(d2, chunk_size), _ceil_div(h2, chunk_size), _ceil_div(w2, chunk_size)

        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            hd0, hh0, hw0 = ii * chunk_size, jj * chunk_size, kk * chunk_size
            hd1, hh1, hw1 = min(hd0 + chunk_size, d2), min(hh0 + chunk_size, h2), min(hw0 + chunk_size, w2)

            ed0, ed1 = max(0, hd0 - half_halo), min(d2, hd1 + half_halo)
            eh0, eh1 = max(0, hh0 - half_halo), min(h2, hh1 + half_halo)
            ew0, ew1 = max(0, hw0 - half_halo), min(w2, hw1 + half_halo)

            # 把扩展后的 1/2 分辨率区域映射回全分辨率输入块; 起点取偶数以保持 stride=2 网格对齐
            qd0 = max(0, 2 * ed0 - 6)
            qh0 = max(0, 2 * eh0 - 6)
            qw0 = max(0, 2 * ew0 - 6)
            qd0 -= qd0 % 2
            qh0 -= qh0 % 2
            qw0 -= qw0 % 2
            qd1 = min(d, 2 * ed1 + 6)
            qh1 = min(h, 2 * eh1 + 6)
            qw1 = min(w, 2 * ew1 + 6)

            y = self.skip_layer1(inp[:, :, qd0:qd1, qh0:qh1, qw0:qw1])
            y = self.skip_layer2(y)

            # q*0 是偶数, 因此局部 1/2 分辨率原点就是 q*0 // 2
            od0, oh0, ow0 = qd0 // 2, qh0 // 2, qw0 // 2
            cd0, ch0, cw0 = hd0 - od0, hh0 - oh0, hw0 - ow0
            cd1, ch1, cw1 = cd0 + (hd1 - hd0), ch0 + (hh1 - hh0), cw0 + (hw1 - hw0)
            out[:, :, hd0:hd1, hh0:hh1, hw0:hw1] = y[:, :, cd0:cd1, ch0:ch1, cw0:cw1]
        return out


    @torch.no_grad()
    def _conv_transpose3d_region(
            self,
            x: Tensor,
            deconv: nn.ConvTranspose3d,
            out_region: Tuple[int, int, int, int, int, int],
            safety: int = 3,
    ) -> Tensor:
        """返回裁剪到指定全分辨率输出区域的 ``deconv(x)``, 避免落地整个反卷积输出。

        专门针对本解码器使用的 stride=2, 输出尺寸恰好为输入两倍的反卷积。
        """
        assert not deconv.training, "_conv_transpose3d_region 只在 eval 模式下有效。"
        stride = _to_3tuple(deconv.stride)
        assert stride == (2, 2, 2), "该辅助函数专用于 stride=2 的解码器反卷积。"
        fd0, fh0, fw0, fd1, fh1, fw1 = out_region
        _, _, d, h, w = x.shape

        id0 = max(0, fd0 // 2 - safety)
        ih0 = max(0, fh0 // 2 - safety)
        iw0 = max(0, fw0 // 2 - safety)
        id1 = min(d, _ceil_div(fd1, 2) + safety)
        ih1 = min(h, _ceil_div(fh1, 2) + safety)
        iw1 = min(w, _ceil_div(fw1, 2) + safety)

        patch = deconv(x[:, :, id0:id1, ih0:ih1, iw0:iw1])
        rd0, rh0, rw0 = fd0 - id0 * 2, fh0 - ih0 * 2, fw0 - iw0 * 2
        rd1, rh1, rw1 = fd1 - id0 * 2, fh1 - ih0 * 2, fw1 - iw0 * 2
        return patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]

    def _run_dec_block1_post(self, patch: Tensor) -> Tensor:
        """反卷积之后的 BN + SiLU (dec_block1 的其余部分)。"""
        y = self.dec_block1[1](patch)
        y = self.dec_block1[2](y)
        return y

    def _run_dec_out_padded(self, patch: Tensor, padlist: Sequence[int]) -> Tensor:
        """在带 padding 的块上执行 dec_out, 并在卷积之间保持人为 halo 为零。"""
        y = self.dec_out[0](patch)
        y = self.dec_out[1](y)
        y = self.dec_out[2](y)
        y = set_pad_as_zero(y, padlist)
        y = self.dec_out[3](y)
        y = self.dec_out[4](y)
        y = self.dec_out[5](y)
        y = set_pad_as_zero(y, padlist)
        y = self.dec_out[6](y)
        y = self.dec_out[7](y)  # Sigmoid
        return y

    @torch.no_grad()
    def _dec_out_chunked(
            self,
            x: Tensor,
            skip_x1: Optional[Tensor],
            inp: Optional[Tensor],
            chunk_size: int,
            halo: int = 3,
    ) -> Tensor:
        """融合并分块的最终解码器。

        避免落地 ``torch.cat([x, skip_x1], dim=1)`` 以及 dec_out 内部的全分辨率中间特征。
        若 ``skip_x1 is None``, 则按需从 ``inp`` 重算。
        """
        assert not self.training, "_dec_out_chunked 只在 eval 模式下有效。"
        if skip_x1 is None:
            assert inp is not None, "按需重算 skip 需要提供输入张量。"
        b, _, d, h, w = x.shape
        out = torch.empty((b, self.out_channels, d, h, w), device=x.device, dtype=x.dtype)
        nd, nh, nw = _ceil_div(d, chunk_size), _ceil_div(h, chunk_size), _ceil_div(w, chunk_size)

        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            (vd0, vh0, vw0, vd1, vh1, vw1), (pd0, ph0, pw0, pd1, ph1, pw1), \
                (rd0, rh0, rw0, rd1, rh1, rw1), padlist = _tile_index_3d(
                ii, jj, kk, chunk_size, (d, h, w), halo
            )
            x_patch = x[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
            if skip_x1 is None:
                skip_patch = self._skip1_region(inp, (pd0, ph0, pw0, pd1, ph1, pw1), halo=2)
            else:
                skip_patch = skip_x1[:, :, pd0:pd1, ph0:ph1, pw0:pw1]
            patch = torch.cat([x_patch, skip_patch], dim=1)
            if sum(padlist) > 0:
                patch = F.pad(patch, padlist, mode="constant", value=0.0)
            patch = self._run_dec_out_padded(patch, padlist)
            out[:, :, vd0:vd1, vh0:vh1, vw0:vw1] = patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]
        return out

    @torch.no_grad()
    def _decode_tail_rank4(self, z_half: Tensor, inp: Tensor, chunk_size: int) -> Tensor:
        """最省内存的尾部解码器。

        把下面这条路径整体融合进分块执行:

            dec_block1[0] (反卷积) -> dec_block1[1:] -> cat(skip_x1) -> dec_out

        这样既不落地全分辨率的 c 通道 dec_block1 输出, 也不落地全分辨率的 c 通道 skip_x1。
        GPU 上只保留一个小块及其 halo, 以及最终输出张量。
        """
        assert not self.training, "_decode_tail_rank4 只在 eval 模式下有效。"
        b, _, dh, hh, wh = z_half.shape
        d, h, w = dh * 2, hh * 2, wh * 2
        out = torch.empty((b, self.out_channels, d, h, w), device=z_half.device, dtype=z_half.dtype)

        dec_out_halo = 3  # dec_out 内三个 3x3x3 卷积
        nd, nh, nw = _ceil_div(d, chunk_size), _ceil_div(h, chunk_size), _ceil_div(w, chunk_size)

        for ii, jj, kk in itertools.product(range(nd), range(nh), range(nw)):
            (vd0, vh0, vw0, vd1, vh1, vw1), dec_patch, dec_crop, dec_padlist = _tile_index_3d(
                ii, jj, kk, chunk_size, (d, h, w), dec_out_halo
            )
            # dec_block1 尾部只有 BN + SiLU, 是逐点算子, 因此这里不需要额外 halo
            deconv_patch = self._conv_transpose3d_region(z_half, self.dec_block1[0], dec_patch, safety=3)
            z1_patch = self._run_dec_block1_post(deconv_patch)

            skip_patch = self._skip1_region(inp, dec_patch, halo=2)
            patch = torch.cat([z1_patch, skip_patch], dim=1)
            if sum(dec_padlist) > 0:
                patch = F.pad(patch, dec_padlist, mode="constant", value=0.0)
            patch = self._run_dec_out_padded(patch, dec_padlist)
            rd0, rh0, rw0, rd1, rh1, rw1 = dec_crop
            out[:, :, vd0:vd1, vh0:vh1, vw0:vw1] = patch[:, :, rd0:rd1, rh0:rh1, rw0:rw1]
        return out


    @torch.no_grad()
    def forward_efficient(self, x: Tensor, rank: int = 1, chunk_size: int = 64) -> Tensor:
        """仅用于推理的优化前向。

        rank = 1: 分块高风险算子/插值, 但保留正常的最终解码器。
        rank = 2: 额外融合并分块最终输出解码器。
        rank = 3: 额外丢弃全分辨率 skip 特征, 在输出阶段按需重算。
        rank = 4: 最省内存路径, 既不落地 skip_x1, 也不落地全分辨率 dec_block1 输出。
        """
        assert not self.training, "rank>0 推理前请先调用 model.eval()。"
        assert rank in (1, 2, 3, 4)

        old_chunk = []
        for m in self.modules():
            if isinstance(m, NearestUpsample3d):
                old_chunk.append((m, m.chunk_size))
                m.chunk_size = chunk_size
        old_stage_chunk = []
        for m in self.modules():
            if isinstance(m, StageModule):
                old_stage_chunk.append((m, m.upsample_chunk_size))
                m.upsample_chunk_size = chunk_size

        try:
            if rank >= 4:
                # 真正的按需路径: skip_x1 从不作为整体张量落地
                skip_x1_for_output = None
                skip_x2 = self._skip2_from_input_chunked(x, chunk_size=chunk_size)
            else:
                skip_x1 = self.skip_layer1(x)
                skip_x2 = self.skip_layer2(skip_x1)
                if rank >= 3:
                    skip_x1_for_output = None
                    del skip_x1
                else:
                    skip_x1_for_output = skip_x1

            z = self._encode_backbone(skip_x2, chunk_size=chunk_size)
            z = torch.cat([z, skip_x2], dim=1)
            del skip_x2

            if rank >= 4:
                return self._decode_tail_rank4(z, x, chunk_size=chunk_size)

            z = conv_transpose3d_chunked(z, self.dec_block1[0], block_size=chunk_size)
            z = self._run_dec_block1_post(z)

            if rank == 1:
                z = torch.cat([z, skip_x1_for_output], dim=1)
                return self.dec_out(z)
            return self._dec_out_chunked(z, skip_x1_for_output, x if rank >= 3 else None, chunk_size=chunk_size)
        finally:
            for m, old in old_chunk:
                m.chunk_size = old
            for m, old in old_stage_chunk:
                m.upsample_chunk_size = old

    def forward(
            self,
            x: Tensor,
            rank: Optional[int] = None,
            chunk_size: int = 64,
    ) -> Tensor:
        if rank is None:
            rank = 0 if self.training else 4
        if rank == 0:
            return self.forward_raw(x)
        return self.forward_efficient(x, rank=rank, chunk_size=chunk_size)


if __name__ == '__main__':
    # 小体积自检; 大体积测试请在 GPU 上用 model.eval() 运行。
    torch.set_num_threads(1)
    model = FaultSegV3(c=8).eval()
    inp = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        out0 = model(inp, rank=0)
        outs = {r: model(inp, rank=r, chunk_size=8) for r in (1, 2, 3, 4)}
    print('input :', tuple(inp.shape))
    print('output:', tuple(out0.shape))
    for r, o in outs.items():
        print(f'rank{r} max abs error:', (out0 - o).abs().max().item())
