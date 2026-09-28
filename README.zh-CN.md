<div align="center">


# FaultSegV3

### Less Is More in Seismic Fault Segmentation



[![PyTorch](https://img.shields.io/badge/PyTorch-3D%20segmentation-EE4C2C?logo=pytorch&logoColor=white)](framework/FaultSegV3.py)
[![Parameters](https://img.shields.io/badge/Parameters-0.563%20M-00897B)](#网络设计)
[![Weights](https://img.shields.io/badge/Pretrained-weights%20included-2563EB)](framework/faultsegv3.ckpt)
[![Data](https://img.shields.io/badge/Data-ModelScope-7356BF)](https://modelscope.cn/datasets/douyimin/CIG-Bench-Dataset)

[English](README.md) · **简体中文**

[核心发现](#核心发现) · [实验结果](#实验结果) · [网络设计](#网络设计) · [快速开始](#快速开始) · [训练](#训练) · [引用](#引用)

Yimin Dou · Xinming Wu · Xiaoming Sun · Runsong Yu  
中国科学技术大学

</div>

---

**地震断层分割网络究竟需要学习什么？** FaultSegV3 从这一问题出发，发现有效提取和保留反射的局部不连续、终止与错断，可能比单纯增加训练数据或模型容量更关键。基于这一认识，我们采用高分辨率三维骨干、有限下采样、充分的数据增强与浅层解码器，在较低资源开销下实现较强的跨工区断层分割能力。

| 紧凑的模型 | 有限的空间压缩 | 合成数据最强基准表现 | 从合成到实际 |
| :---: | :---: | :---: | :---: |
| **0.563 M** 参数 | 主路径仅 **1 次**下采样 | 平均 Tol-IoU@2 **0.7376** | **不使用实际工区数据训练** |

> **本文的贡献不限于轻量化网络：** 通过数据规模与模型容量实验提出局部特征假设，以网络设计验证其实用价值，并构建实际工区参考标签，支持统一协议下的定量比较。

<p align="center">
  <a href="docs/assets/study-overview.jpg"><img src="docs/assets/study-overview.jpg" alt="研究总览：浅层特征、数据规模、模型容量及精度与资源对比" width="100%"></a>
  <br><sub>从“网络学到了什么”出发，连接实验发现与网络设计。点击图片可查看高清原图。</sub>
</p>

## 核心发现

### 1. 很少的合成数据，已经可以支持实际工区泛化

仅使用 **2 个 128³ 合成地震体**训练，FaultSegV3 就能在实际工区中识别主要断层。使用 5 个训练体时，最优域平衡 Tol-IoU@2 达到 **0.5647**，约为数据规模实验中最高观测峰值 **0.6365 的 89%**。从 5 个增加到 200 个训练体，源数据量扩大 **40 倍**，分数的绝对增益为 **0.0668**。

<p align="center">
  <a href="docs/assets/few-volume-generalization.jpg"><img src="docs/assets/few-volume-generalization.jpg" alt="完整展示两个和五个合成训练体，以及同一组实际工区视角下的泛化结果" width="100%"></a>
  <br><sub>上方展示全部合成训练体，下方展示实际工区预测；实际数据未用于训练。图中采用第 80,000 步的参数 EMA 权重。</sub>
</p>

这一结果说明，**重复增加同一生成机制下的合成样本会出现收益递减**，并不否定新增地质多样性的价值。“两体训练”是独立的少样本实验，下方最终方法比较使用了更完整的合成训练集。

### 2. 更大的网络，不一定带来更好的断层分割

HRNet 参数量从 **77.3 M 增至 214.6 M**，扩大 **2.78 倍**后，按论文分数平滑协议计算的实际工区 Tol-IoU@2 峰值仅增加约 **0.008**。更宽的模型还可能在持续训练后出现更明显的性能回落。

### 3. 应优先保留细尺度空间信息

浅层特征中已可观察到断层响应，后续层更像是在抑制非断层响应、降低噪声并改善连续性。FaultSegV3 因此保留全分辨率跳跃连接与持续的高分辨率处理，同时用辅助分支补充上下文。

**结论的适用范围：** 这些证据支持当前逐体素判别学习范式下的设计假设，并不证明全局地质信息无用，也不意味着小模型在所有任务中都更好。

<details>
<summary><b>展开查看数据规模与模型容量实验</b></summary>

**数据规模：1、2、5、10、50、100、200 个合成训练体。** 综合分数对合成域均值与实际域均值各赋予 1/2 权重。

![HRNet、ResUNet 与 FaultSegV3 的数据规模实验](docs/assets/data-scaling.jpg)

**模型容量与训练动态。** 曲线对 F3、USGS1、USGS2 的平均分数做 EMA 平滑（α = 0.25）；这里是分数曲线平滑，与模型参数 EMA 不同。

![模型容量、峰值与最终性能、训练稳定性对比](docs/assets/capacity-scaling.jpg)

</details>

## 实验结果

### 性能亮点

论文报告的性能指标均为 **5 个评价体的算术平均值**：2 个合成体、USGS1、USGS2，以及**不含多边形断层的 F3 评价体**，每个体的权重均为 1/5。

相较 CIG-Bench predictor，FaultSegV3 的 **Tol-IoU@2 提高 0.0886（相对提升 13.7%）**，参数量约为其 **1/61**，标准前向 MACs 减少约 **73%**。在参与比较的方法中，FaultSegV3 的平均 Tol-IoU@2、Accuracy 与 Recall 最高；FaultSSL 的 Precision 最高，FaultNet 的资源开销最低。

<details>
<summary><b>评价协议与数字解读</b></summary>

- **容差指标：** 预测阈值为 0.5，以三维欧氏距离不超过 2 个体素的邻域进行匹配，共 33 个整数偏移。Precision 与 Recall 分别统计匹配到的预测和标签体素。Tol-IoU@2 由容差 F1 经 `F1 / (2 − F1)` 得到，不是严格的逐体素 Jaccard 交并比；Accuracy 同样基于容差匹配计数。
- **训练数据：** 最终方法比较中的 FaultSegV3 使用 200 个 FaultSegV1 合成体和 CIG-Bench 的 `struc_00.npz`，没有实际工区训练数据。其他方法使用可获得的预训练权重，因此比较对象是方法与权重的组合，并非统一重训条件下的纯架构消融。
- **计算量：** MACs 对应 128³ 输入的标准前向，仅统计卷积与转置卷积；1 MAC = 2 FLOPs，不代表实测运行时间。
- **输入容量：** 在 30 GiB PyTorch 分配器预算内，以边长步进 16 搜索 batch=1、FP16 推理可处理的立方体。FaultSegV3 使用 rank-4 算子分块，块大小为 64。该测试不包含预处理、指标计算和 CUDA 上下文开销，也不代表训练容量或统一 FP16 条件下重新计算的精度。
- **不同实验的权重不同：** 数据规模实验对合成域和实际域各赋权 1/2；最终方法比较对 5 个体各赋权 1/5，两种综合分数不应混为同一指标汇总口径。

</details>

### 实际工区：细小断层与密集断层网络

**F3 多边形断层。** 在下方补充视角中，FaultSegV3 展示出更密集的细尺度断层响应。

![F3 多边形断层的多方法定性比较](docs/assets/field-f3.jpg)

**子图说明：** (a, i) 原始地震；(b) Petrel；(c) FaultSegV1；(d) FaultNet；(e) FaultSegV2；(f, j) FaultSSL；(g, k) CIG-Bench；**(h, l) FaultSegV3**。这里的 F3 多边形断层体**仅用于定性比较**，与论文定量评价使用的 F3 体不同。

<details>
<summary><b>USGS1 · 分支断层与细小连接</b></summary>

![USGS1 实际工区比较](docs/assets/field-usgs1.jpg)

</details>

<details>
<summary><b>USGS2 · 穿过反射扰动区间的断层覆盖</b></summary>

![USGS2 实际工区比较](docs/assets/field-usgs2.jpg)

</details>

<details>
<summary><b>USGS3 · 密集交叉断层</b></summary>

![USGS3 实际工区比较，仅用于定性评价](docs/assets/field-usgs3.jpg)

</details>

<details>
<summary><b>NLOG · 主断层与短小连接</b></summary>

![NLOG 实际工区比较，仅用于定性评价](docs/assets/field-nlog.jpg)

</details>

所有工区采用相同的子图排列。USGS3 与 NLOG 仅参与定性比较。图中差异主要体现在连续性、覆盖范围与响应厚度；更密集的响应本身不能证明所有连接都是真实断层。

## 网络设计

![FaultSegV3 架构：全分辨率跳跃连接、双分辨率处理、六个融合模块与浅层解码器](docs/assets/architecture.jpg)

| 设计 | 实现 | 作用 |
| :--- | :--- | :--- |
| 保留细节 | 全分辨率跳跃特征，主路径仅一次下采样 | 保留弱反射不连续、终止与微小错断 |
| 适度引入上下文 | 半分辨率主分支 `c=8`，四分之一分辨率辅助分支 `2c=16` | 区分断层、噪声和非断层振幅边缘 |
| 反复交换信息 | 六个双向融合模块，每个模块的每条分支包含四个残差块 | 结合周围信息逐步细化局部线索 |
| 浅层解码 | 一次恢复至全分辨率，并融合跳跃特征 | 利用保留的空间特征重建薄层结构 |
| 促进泛化 | 在线几何与振幅增强，最高 0.1 的 Drop Path | 减少对训练源特有外观的依赖 |

**“一次下采样”指主特征路径**；四分之一分辨率辅助分支另有下采样。当前 `c=8` 实现包含 **562,625 个可训练参数**。

推理额外提供 **`rank=0…4` 算子分块**与显存不足时的自动空间分块。这里的 `rank` 表示内存优化级别，与张量秩或分布式进程编号无关。

## 快速开始

### 安装

建议使用 **Python 3.10+**。请从 [PyTorch 官网](https://pytorch.org/get-started/locally/)选择适合硬件的 PyTorch 安装版本。

```bash
git clone https://github.com/douyimin/FaultSegV3.git
cd FaultSegV3

# 基本推理依赖
python -m pip install torch numpy

# 自动下载与三维交互可视化
python -m pip install modelscope cigvis PyQt5

# 训练额外依赖
python -m pip install torchvision pytorch-lightning kornia matplotlib tensorboard
```

训练时需确保 `torchvision` 与 PyTorch 版本匹配。交互可视化需要桌面与 OpenGL 环境；文件推理无需安装可视化依赖。预训练权重已包含在 [`framework/faultsegv3.ckpt`](framework/faultsegv3.ckpt)。

### 一条命令运行 F3 演示

```bash
python inference.py
```

程序从 [ModelScope](https://modelscope.cn/datasets/douyimin/CIG-Bench-Dataset) 下载或读取缓存的 `FaultMetricData/F3-GT.npz`，选取 `seis_polyfault`，将**三个轴均放大 2 倍**后推理，再恢复原始尺寸，保存为 `F3-GT-faultsegv3-prediction.npz`，并打开三维地震与预测对照视图。放大后内存需求会增加，建议使用 CUDA GPU 运行该演示。

### 推理自己的地震体

输入应为有限数值构成的三维 NumPy 数组，轴顺序为 **`(t, h, w)`**：时间/深度在第一维，后两维为横向空间轴。支持 `.npy` 与 `.npz`。

```bash
# 文件推理，不打开图形界面
python inference.py seismic.npy prediction.npz --resize-back

# 指定 NPZ 内的数组，并显示结果
python inference.py survey.npz prediction.npz --input-key seis --resize-back --visualize

# 小体积 CPU 推理
python inference.py seismic.npy prediction.npz --device cpu --resize-back
```

输出 NPZ 包含 `prediction` 和 `input`。默认**将小于或等于 0.5 的预测置零，其余仍为概率值，并非二值标签**；使用 `--threshold 0` 保留完整的非负概率图。`--resize-back` 将输出恢复到原始尺寸；不加该参数时，输入会被重采样为各维均为 16 倍数的尺寸。

### Python 接口

```python
import numpy as np
from inference import FaultPredictor

seismic = np.load("seismic.npy")  # (t, h, w)
predictor = FaultPredictor(device="cuda")
probability, used_seismic = predictor.predict(
    seismic,
    threshold=0.0,
    resize_back=True,
    rank=4,
    chunk_size=64,
)
fault_mask = probability > 0.5
np.savez_compressed("prediction.npz", prediction=probability, fault=fault_mask)

# 可选：桌面交互显示
# predictor.visualize(used_seismic, probability)
```

<details>
<summary><b>显存控制与常用推理参数</b></summary>

| 参数 | 默认值 | 含义 |
| :--- | :--- | :--- |
| `--weights` | `framework/faultsegv3.ckpt` | 本地模型权重 |
| `--device` | `cuda` | CUDA 设备或 CPU；CUDA 不可用时回退至 CPU |
| `--rank` | `4` | `0` 为标准前向；`1–4` 为逐级增加的算子内存优化 |
| `--chunk-size` | `64` | 内部算子分块尺寸 |
| `--scale` | `1.0` | 统一缩放；非 1 值覆盖分轴缩放设置 |
| `--scale-t / --scale-h / --scale-w` | `1.0` | 各轴独立缩放 |
| `--resize-back` | 关闭 | 用最近邻插值恢复原尺寸 |
| `--gpu-memory-fraction` | `0.9` | 相对初始空闲显存的检测预算 |
| `--fallback-halo` | `16` | 回退分块的上下文重叠宽度 |
| `--no-oom-fallback` | 关闭 | 禁用显存不足时的自动空间分块重试 |

回退策略仅分割横向轴，保留完整时间/深度轴。由于各块独立归一化，分块结果可能与整体推理存在边界差异。显存检测可能在一次前向完成后触发重试，不是硬性的显存预留机制。

```bash
python inference.py seismic.npy prediction.npz --rank 4 --chunk-size 32 --resize-back
python inference.py --help
```

</details>

## 训练

训练流程提供数据自动准备、按 batch 进行多尺度采样、数据转移至设备后的增强、Dice 损失、AdamW、余弦学习率衰减、可选 EMA，以及可恢复训练状态的 Lightning 检查点。

```bash
# 仅下载并解压 FaultSegV1
python train.py --dataset v1 --data-root ./data --prepare-only

# 从附带权重继续训练，显式指定较小 batch
python train.py --dataset v1 --data-root ./data --size 2 128 128 128 --workers 0

# 随机初始化，从头训练
python train.py --dataset v1 --data-root ./data --no-weights --size 2 128 128 128 --workers 0

# 多尺度训练：每个尺度独立设置 batch 大小
python train.py --dataset all --data-root ./data --size 2 128 128 128 --size 1 256 176 176 --workers 0

# 恢复优化器、调度器及完整训练状态
python train.py --dataset v1 --data-root ./data --resume model_weights/last.ckpt --size 2 128 128 128 --workers 0
```

`--size` 接受 **`BATCH D H W`** 四个数，可重复指定。请根据 GPU 调整 batch；`--workers 0` 便于起步，之后可增加工作进程提高吞吐。数据集支持 `v1`、`v2` 和 `all`（V1 + V2）。训练 NPZ 需包含成对的 `seis` 与 `fault` 数组。

| 设置 | 当前脚本默认值 |
| :--- | :--- |
| 初始化 | 加载附带的预训练权重；从头训练使用 `--no-weights` |
| 优化器 / 损失 | AdamW / masked Dice |
| 学习率 | `1e-4 → 5e-6`，余弦衰减终点为 100,000 步 |
| 精度 | `bf16-mixed` |
| EMA | 默认开启，衰减 `0.9996`；用 `--no-ema` 关闭 |
| 检查点 | 每 2,500 步保存至 `model_weights/` |
| 日志 / 预览 | TensorBoard 日志位于 `logs/`；训练预览位于 `results/` |

**复现论文时需区分配置：** 上述默认设置服务于便捷使用，并非论文实验的完整复现协议。规模实验使用 FaultSegV1 指定子集、随机初始化、80,000 步训练，定量评价使用普通检查点。最终比较使用 200 个 FaultSegV1 体加 `struc_00.npz`，而 `--dataset all` 加载 V1 + V2。当前代码快照未包含完整的规模实验/基线评价入口，也未提供训练步数上限参数；复现时需设置对应子集及 `Trainer(max_steps=80000)`。`--cosine-steps` 仅控制学习率衰减，不控制停止训练。脚本内部设置了 `CUDA_VISIBLE_DEVICES="0,1"`，如需调整可见 GPU，请修改这一行。

## 数据与实际工区参考标签

数据托管于 [ModelScope 的 CIG-Bench 数据集](https://modelscope.cn/datasets/douyimin/CIG-Bench-Dataset)。代码自动准备 `ClassicDatasets/FaultSegV1.zip` 和/或 `ClassicDatasets/FaultSegV2.zip`，优先复用本地缓存。

本文为 **F3、USGS1 和 USGS2** 构建了经过校正与细化的实际工区参考标签：

```mermaid
flowchart LR
    A[自动断层预测] --> B[GEM 交互校正与缺失补全]
    B --> C[OSV 方向扫描与细化]
    C --> D[清理孤立伪响应与非断层碎片]
    D --> E[实际工区参考标签]
```

这些标签使不同方法能够在共同参考下接受定量评价。论文合成测试体使用 **`seis_noised`** 数组；F3 交互演示使用 **`seis_polyfault`**。README 配图与论文原图的对应关系见[图片来源说明](docs/assets/README.md)。

## 项目结构

```text
FaultSegV3/
├── inference.py                 # 本地权重、推理、显存回退与三维演示
├── train.py                     # 数据准备与 Lightning 训练入口
├── framework/
│   ├── FaultSegV3.py            # 高分辨率骨干与算子分块推理
│   ├── framework.py             # Lightning 模块、损失及优化器
│   ├── custom_callback.py       # EMA、权重导出与训练预览
│   └── faultsegv3.ckpt          # 附带预训练权重
├── seisDataset/
│   ├── dataset.py              # 下载缓存、成对加载与多尺度采样
│   └── gpu_aug.py              # 几何与振幅增强
├── docs/assets/                 # README 横幅与论文原图
├── README.md                    # 英文主页
└── LICENSE-CIG-Bench            # 改编自 CIG-Bench 部分的许可证
```

## 引用

如果本工作对你的研究有帮助，请引用论文。下方条目仅采用已确认的稿件信息，不填写尚未核实的期刊、DOI 或出版信息。

```bibtex
@misc{dou_faultsegv3,
  title  = {{FaultSegV3}: Less Is More in Seismic Fault Segmentation},
  author = {Dou, Yimin and Wu, Xinming and Sun, Xiaoming and Yu, Runsong},
  note   = {Manuscript},
  url    = {https://github.com/douyimin/FaultSegV3}
}
```

联系：[Yimin Dou](mailto:douyimin@ustc.edu.cn) · [Xinming Wu](mailto:xinmwu@ustc.edu.cn)。

**致谢。** 感谢 FaultSeg3D/FaultSegV1、FaultSeg3D+/FaultSegV2、HRNet、FaultNet、FaultSSL、CIG-Bench、GEM、OSV 及公开地震工区数据的贡献者。可视化使用 [cigvis](https://github.com/JintaoLee-Roger/cigvis)。改编自 CIG-Bench 的推理部分遵循 [`LICENSE-CIG-Bench`](LICENSE-CIG-Bench)；该文件对应这些改编部分，不代表整个仓库的统一许可声明。
