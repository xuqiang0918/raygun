"""Raygun 单样本推理：性能 + 峰值显存（SDAA / CUDA 两侧同脚本）。

用法（容器内，SDAA）：
    SDAA_VISIBLE_DEVICES=0 /home/py312/bin/python3 sdaa/scripts/bench_single.py sdaa
用法（A100，CUDA）：
    CUDA_VISIBLE_DEVICES=0 python sdaa/scripts/bench_single.py cuda

输出：weight-load 时间 / 端到端耗时(多轮取中位数) / 峰值显存 / 生成序列指纹。
"""

import hashlib
import json
import os
import pathlib
import statistics
import sys
import time

import torch


def seq_fp(s):
    """稳定指纹（PYTHONHASHSEED 无关），用于跨平台逐位对比。"""
    return hashlib.md5(s.encode()).hexdigest()[:16]

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEV = sys.argv[1] if len(sys.argv) > 1 else "sdaa"
CKPT = os.environ.get("RAYGUN_CKPT", "")
WARMUP = int(os.environ.get("RAYGUN_WARMUP", "2"))
REPEAT = int(os.environ.get("RAYGUN_REPEAT", "5"))
NOISE = float(os.environ.get("RAYGUN_NOISE", "1.0"))
GENLEN = int(os.environ.get("RAYGUN_GENLEN", "175"))

SEQ = os.environ.get("RAYGUN_SEQ", "") or (
    "MVSKGEEDNMAIIKEFMRFKVHMEGSVNGHEFEIEGEGEGRPYEGTQTAKLKVTKGGPLPFAWDILSPQFMYGSKAYVKHPADIPDY"
    "LKLSFPEGFKWERVMNFEDGGVVTVTQDSSLQDGEFIYKVKLRGTNFPSDGPVMQKKTMGWEASSERMYPEDGALKGEIKQRLK"
    "LKDGGHYDAEVKTTYKAKKPVQLPGAYNVNIKLDITSHNEDYTIVEQYERAEGRHSTGGMDELYK")


def accel_sync(dev):
    if dev == "sdaa":
        torch.sdaa.synchronize()
    else:
        torch.cuda.synchronize()


def peak_mem(dev):
    if dev == "sdaa":
        return torch.sdaa.max_memory_allocated() / 1024 ** 3
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def main():
    device = torch.device(DEV)
    print("[%s] torch %s | device %s | ckpt %s" % (
        DEV, torch.__version__, device, os.path.basename(CKPT) or "<random-init>"))

    from esm.pretrained import esm2_t33_650M_UR50D
    from raygun.pretrained import load_raymodel

    t0 = time.time()
    esmmodel, esmalphabet = esm2_t33_650M_UR50D()
    if CKPT:
        ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
        rayl = load_raymodel(ckpt, fixed_batching=False)
        missing = unexpected = None
        del ckpt
    else:
        from raygun.modelv2.raygun import Raygun as RaygunV2
        from raygun.modelv2.ltraygun import RaygunLightning
        rayl = RaygunLightning(RaygunV2(numencoders=12, numdecoders=12,
                                        fixed_esm_batching=False), esmmodel)
        missing = unexpected = None
    t_load = time.time() - t0

    esmmodel = esmmodel.to(device).eval()
    raymodel = rayl.model.to(device).eval()
    bc = esmalphabet.get_batch_converter()

    _, _, tokens = bc([("prot", SEQ)])
    tokens = tokens.to(device)
    print("seq len: %d" % (tokens.shape[1] - 2))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    else:
        torch.sdaa.reset_peak_memory_stats()

    def one_pass():
        with torch.no_grad():
            emb = esmmodel(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            emb = emb[:, 1:-1, :]
            encoder = raymodel.encoder(emb, noise=NOISE)
            out = raymodel.get_sequences_from_fixed(encoder, GENLEN)[0]
        return out

    for _ in range(WARMUP):
        one_pass()
    accel_sync(DEV)

    times = []
    last = None
    for _ in range(REPEAT):
        accel_sync(DEV)
        t0 = time.time()
        last = one_pass()
        accel_sync(DEV)
        times.append(time.time() - t0)

    peak = peak_mem(DEV)
    med = statistics.median(times)

    print("weight load          : %.1f s" % t_load)
    print("e2e median (%d runs) : %.3f s   (min %.3f / max %.3f)" % (
        REPEAT, med, min(times), max(times)))
    print("peak memory          : %.2f GiB" % peak)
    print("gen len / fingerprint: %d / %s" % (len(last), seq_fp(last)))
    print("generated seq        : %s" % last)

    summary = {
        "device": DEV, "torch": torch.__version__,
        "ckpt": os.path.basename(CKPT), "noise": NOISE, "genlen": GENLEN,
        "weight_load_s": round(t_load, 1),
        "e2e_median_s": round(med, 4),
        "e2e_min_s": round(min(times), 4),
        "e2e_max_s": round(max(times), 4),
        "peak_mem_gib": round(peak, 2),
        "gen_len": len(last),
        "seq_fingerprint": seq_fp(last),
        "seq": last,
    }
    out = pathlib.Path(os.environ.get("RAYGUN_OUT", "/tmp/raygun_results")) / ("bench_%s.json" % DEV)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("written: %s" % out)


if __name__ == "__main__":
    main()
