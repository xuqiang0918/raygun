# Raygun SDAA 适配说明

> 仓库：https://github.com/rohitsinghlab/raygun （适配分支：`xuqiang0918/raygun` → `adapt/sdaa`）
> 基线 commit：`a74d7971650be4ebe230dc4664e5f5356aac568c`
> 目标平台：太初 SDAA（loongarch64，Torch-SDAA 3.2.1 / PyTorch 2.12.0）
> 对照平台：CPU（同容器）；CUDA A100（torch 2.4.1+cu121，§5 已对照）

---

## 1. 结论

Raygun 是纯 PyTorch 实现：**无自定义 CUDA/Triton/OpenCL 算子**（编译产物 0 个、`.cuda()` 0 处），
适配工作量集中在对加速器的**设备选择**上。

| 项 | 结论 |
| --- | --- |
| 改动文件数 | 4 |
| 改动量 | 81 增 / 13 删 |
| 自定义算子移植 | 无 |
| 算子回落 | 无（见 §5） |
| 官方 CLI | `raygun-sample-single` / `raygun-sample-multiple` 均可用 |
| 精度对齐 | 确定性前向最大绝对差 2.28e-02（相对 4.6e-04，见 §7） |

改动分两类：

- **`[upstream]` 通用兼容性修复 1 处** —— `h5py` 可选依赖改为惰性导入。
- **`[sdaa-adapt]` 平台适配 2 处** —— 设备选择归一化、Lightning 训练器后端参数化。

### 最终验收指标

| 指标 | 结果 |
| --- | --- |
| 端到端 | 官方 CLI 真权重跑通：EGFP（236 aa → 175）与 lacZ（1024 aa → 950）两组，`--device sdaa:0`，产出候选/PLL/过滤文件齐全 |
| 峰值显存 | EGFP 5.67 GiB / lacZ 5.85 GiB（CUDA 侧 5.65 / 5.85 GiB，同水平） |
| 数值 | 确定性前向逐元素最大绝对差（SDAA vs CPU）：embedding 1.2e-05、encoder 输出 3.3e-04、logits 2.3e-02 |
| 精度 | ATLAS_valid 139 条批量生成：131 条逐位完全一致，top-3/top-5 一致率 100%，JS 散度中位 2.2e-07 bits；真实采样序列官方 PLL 置信度分布与 CUDA 无显著差异（MWU p=0.478）；单条 logits 相对差 4.6e-04 / 7.5e-04（EGFP/lacZ），与 CUDA 侧同量级 |
| 性能 | SDAA/CUDA：EGFP 10.7×（1.484 s vs 0.139 s）、lacZ 16.7×（4.547 s vs 0.273 s）、ATLAS_valid 批量 6.3×（median 0.648 s vs 0.104 s） |
| 算子回落 | 0（10529 次 aten 调用，唯一 accel→cpu 为写盘前显式搬运，见 §5） |

---

## 2. 环境

容器（`3.2.1-torch_sdaa3.2.1`）内依赖除一项外全部现成：torch / torch_sdaa、fair-esm 2.0.1、
einops / biopython / pandas / numpy / tqdm、torchmetrics / hydra / omegaconf / lightning。
`h5py` 缺失，但主路径不需要（见 §3.1）。raygun 未发布到 PyPI，从仓库安装：`pip install -e . --no-deps`。

权重：ESM-2 650M（`esm2_t33_650M_UR50D`）由 fair-esm 自动下载；
Raygun `2.2M` / `4.4M` / `8.8M` 来自 Zenodo record 15447158 / 15578824 / 17253788，
三者体积相同（约 3.33 GB），`--model` 默认 `8.8M`。

---

## 3. 改动

### 3.1 `[upstream]` h5py 惰性导入

`raygun/modelv2/loader.py`：h5py 仅预计算 embedding 路径使用，原模块级 `import h5py`
导致无 h5py 环境启动即挂（NVIDIA 上同样会挂）。将 import 移入 `collatefn` 中
`self.precomputed` / `self.save` 两个实际使用分支。

### 3.2 `[sdaa-adapt]` 设备归一化

`raygun/commands/generate_samples_single.py`、`generate_samples_multiple.py`：
`--device` 由 `int` 改 `str`（默认 `"0"`），新增 `resolve_device()`：

| 输入 | 输出 |
| --- | --- |
| `-1` / `cpu` | `cpu` |
| `0`, `1`, ... | `sdaa:N`（SDAA 可用）/ `cuda:N` |
| `sdaa:N` / `cuda:N` | 原样 |

SDAA 上 `torch.cuda.is_available()` 恒为 `False`，原 int 类型无法表达 SDAA 设备。

### 3.3 `[sdaa-adapt]` Lightning Trainer 后端参数化

`raygun/modelv2/training.py`：`finetune` 分支 `L.Trainer(accelerator="gpu", devices=1)`
硬编码改为参数传入，默认值与原行为一致。训练路径不在验收主线内，未实测训练。

---

## 4. 运行

```bash
# 单序列生成
raygun-sample-single <template.fasta> <out_dir> \
    --minlength 175 --maxlength 199 \
    --model 2.2M --device sdaa:0 \
    --num_raygun_samples_to_generate 500

# 多序列生成
raygun-sample-multiple <templates.fasta> <out_dir> \
    --lengthinfo <lengthinfo.json> --device sdaa:0
```

`--device` 也可写 `0`（自动解析为 `sdaa:0`）或 `cpu`。

Python API：

```python
from raygun.pretrained import raygun_2_2mil_800M
model = raygun_2_2mil_800M().to("sdaa")
```

---

## 5. 算子覆盖统计

方法：`TorchDispatchMode` 拦截推理链路中每一次 aten 调用，记录输入/输出张量的设备配对并归类
（同一脚本 `sdaa/scripts/check_op_fallback.py` 两侧各跑一次）。判据：PyTorch 不会把加速器
张量的算子静默放到 CPU 上执行再搬回来，因此「输入含加速器张量、输出不含」是回落的唯一形态。

| 类别 | SDAA (torch 2.12.0) | CUDA A100 (torch 2.4.1) |
| --- | --- | --- |
| total aten calls | 10529 | 11456 |
| distinct ops | 54 | 53 |
| accel-internal | 9451 | 10377 |
| cpu→accel | 244 | 244 |
| **accel→cpu** | **1** | **1** |
| cpu-only | 442 | 443 |
| host-sync | 391 | 391 |

唯一的 `accel→cpu` 项为 `aten._to_copy.default`（1 次），是生成结果写盘前的显式
`.cpu()` 搬运，不是算子回落，两侧形态一致。`cpu-only` 的 13 项（`add` / `eq` / `mul` /
`floor_divide` / `remainder` / `select` 等）均为序列解码循环里的标量长度计算，与设备无关。
**结论：零算子回落。**

---

## 6. 性能

单进程单卡，`--model 2.2M`（真权重），warmup 2 次后测 5 轮取中位（单样本数值为
热身后 3 个独立进程复测的中位，首次 CUDA 测量受冷缓存影响偏高已弃用）：

- 单样本：EGFP（236 aa → 175）SDAA 1.484 s / CUDA 0.139 s，**10.7×**；
  lacZ（1024 aa → 950）SDAA 4.547 s / CUDA 0.273 s，**16.7×**；峰值显存两侧 5.7–5.8 GiB。

批量数据集：**ATLAS_valid**（`data/fastas/ATLAS_valid.fasta`，139 条模板，53–839 aa，
每条生成 0.75×长度，每条 1 次 warmup + 3 次计时取中位，两侧同脚本 `batch_eval.py`）：

| 项 | SDAA（1 卡） | CUDA（1×A100-40G） |
| --- | --- | --- |
| 每条耗时中位数 | 0.648 s | 0.104 s |
| 每条耗时均值 / p90 | 0.714 s / 1.022 s | 0.106 s / 0.115 s |
| 139 条总耗时（各条中位之和） | 99.3 s | 14.7 s |
| 峰值显存 | 5.79 GiB | 5.76 GiB |

倍率 **6.3×**（0.6482 / 0.1035）。

---

## 7. 精度

数值：确定性前向（`noise=0`）单条 236 aa，SDAA vs CPU 逐元素最大绝对差
embedding 1.2e-05 / encoder 输出 3.3e-04 / logits 2.3e-02（logits absmax 49.15，
相对量 4.6e-04）；CUDA vs CPU 同量级。ckpt 键核对按官方层级（`RaygunLightning`，
`strict=False`）：`unexpected = 0`，`missing = 572` 全为 `esmmodel.*`（ESM-2 预训练权重，预期）。
脚本：`sdaa/scripts/check_numeric.py` / `check_ckpt_and_accuracy.py`（两侧同脚本）。

### 批量生成一致率（ATLAS_valid 139 条，SDAA vs CUDA 直接对照）

noise=0 确定性生成下，logits 经 softmax 即逐位置氨基酸采样分布，argmax 即实际生成的
氨基酸序列。指标（`sdaa/scripts/batch_eval.py` 落盘 + `compare_batch.py` 比对）：

| 指标 | 结果 |
| --- | --- |
| 生成序列逐位完全一致 | **131 / 139 条** |
| top-1 一致率 | 均值 99.976%，最低 99.40%（未完全一致的 8 条各仅 1–2 个位置分歧） |
| top-3 / top-5 一致率 | **100% / 100%**（所有分歧位置的 SDAA 首选氨基酸均在 CUDA top-3 内） |
| 采样分布 JS 散度 | 中位 2.2e-07 bits，最大 2.2e-06 bits（log2 域） |
| logits 余弦相似度 | ≥ 0.9999996 |
| mean 相对差 | ≤ 2.4e-03 |

结论：跨平台确定性生成的序列在 139 条中 131 条逐位相同，其余 8 条每条仅 1–2 个位置
偏差且该位置候选集（top-3）仍一致——采样分布层面 SDAA 与 CUDA 无行为差异。

### 生成质量置信度（官方 PLL，noise=1.0 真实采样）

官方候选过滤置信度是 PLL（`raygun/pll.py::get_PLL`，ESM-2 log-likelihood 按
`/|−0.406L+1.363|` 长度归一）。两侧各对 ATLAS_valid 139 条模板真实采样（`noise=1.0`），
汇总到同一台 CUDA 机用同一份 ESM-2 统一打分：

| 指标（长度归一 PLL） | SDAA 生成序列 | CUDA 生成序列 |
| --- | --- | --- |
| n | 139 | 139 |
| 均值 ± 标准差 | −1.197 ± 0.136 | −1.203 ± 0.154 |
| 中位数 | −1.207 | −1.223 |
| min / max | −1.633 / −0.854 | −1.563 / −0.736 |
| p5 ~ p95 | −1.390 ~ −0.972 | −1.448 ~ −0.945 |
| p25 ~ p75 | −1.272 ~ −1.123 | −1.294 ~ −1.106 |

Mann-Whitney U 检验 p = 0.478，Kolmogorov-Smirnov 检验 p = 0.579 —— 两侧生成序列的
置信度分布**无显著差异**，SDAA 生成质量与 CUDA 相当。

注：`noise=1.0` 采样消耗设备随机数，两平台 RNG 不同，生成序列不作逐位对比；
CUDA 侧 torch 2.4.1+cu121 / fair-esm 2.0.0，SDAA 侧 torch 2.12.0 / fair-esm 2.0.1。

---

## 快速验证

前提：容器 `3.2.1-torch_sdaa3.2.1` 内先 `set +u; source /opt/tecoai/setvars.sh`；
权重路径由环境变量 `RAYGUN_CKPT` 传入；所有脚本第一个位置参数为设备（`sdaa` / `cuda` / `cpu`），
两侧同脚本。

| 脚本 | 用途 | 怎么跑 | 产出 |
| --- | --- | --- | --- |
| `bench_single.py` | 单样本耗时 + 峰值显存（内置 EGFP 236aa 输入） | `RAYGUN_CKPT=<ckpt> python3 bench_single.py sdaa` | `bench_sdaa.json`（e2e 中位 / peak） |
| `batch_eval.py` | ATLAS_valid 139 条批量性能 + noise=0 确定性 logits | `RAYGUN_CKPT=<ckpt> RAYGUN_FASTA=data/fastas/ATLAS_valid.fasta RAYGUN_OUT=<out> python3 batch_eval.py sdaa` | `batch_sdaa.json` + `batch_sdaa_logits.npz` |
| `compare_batch.py` | 两侧 logits 比对（top-k 一致率 / JS 散度 / cos） | `python3 compare_batch.py <out>/batch_sdaa_logits.npz <cuda侧batch_cuda_logits.npz>` | 终端打印 + json |
| `gen_pll.py` | 真实采样（noise=1.0）生成 + 官方 PLL 打分 | 生成：`RAYGUN_CKPT=<ckpt> RAYGUN_FASTA=<fasta> RAYGUN_OUT=<out> python3 gen_pll.py gen sdaa`；打分：`RAYGUN_IN=<fasta> RAYGUN_TAG=sdaa python3 gen_pll.py pll <dev>` | `gen_sdaa.fasta` + `pll_sdaa.json` |

CUDA 基线用同一脚本传 `cuda` 生成（§6/§7 数字即来源于此）。

---

## 交付物

| 路径 | 说明 |
| --- | --- |
| `sdaa/SDAA_ADAPTATION.md` | 本文档 |
| `sdaa/patches/0001-upstream-compat-Raygun.patch` | `[upstream]` 改动 |
| `sdaa/patches/0002-sdaa-adapt-Raygun.patch` | `[sdaa-adapt]` 改动 |
| `sdaa/scripts/bench_single.py` | ① 单样本耗时与峰值显存 |
| `sdaa/scripts/batch_eval.py` | ② ATLAS_valid 批量评测（性能 + noise=0 logits 落盘） |
| `sdaa/scripts/compare_batch.py` | ③ 批量 logits 比对（top-k 一致率 / JS 散度 / cos） |
| `sdaa/scripts/gen_pll.py` | ④ 真实采样生成 + 官方 get_PLL 置信度打分 |
