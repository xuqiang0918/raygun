# -*- coding: utf-8 -*-
"""Raygun ATLAS_valid 批量评测（SDAA/CUDA/CPU 两侧同脚本）。

用法:
    python3 batch_eval.py <dev>     # dev: sdaa / cuda / cpu
env:
    RAYGUN_CKPT   2.2M 权重路径（必填）
    RAYGUN_FASTA  评测 fasta（必填，如 data/fastas/ATLAS_valid.fasta）
    RAYGUN_OUT    输出目录
    RAYGUN_SKIP_PERF=1  跳过性能臂（CPU 对照臂用）
输出:
    $RAYGUN_OUT/batch_<dev>.json           每条 seqlen/genlen/耗时(3 次中位)
    $RAYGUN_OUT/batch_<dev>_logits.npz     每条 noise=0 确定性 logits (genlen,32)
"""
import os, sys, json, time, pathlib
import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else "sdaa"
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FASTA = os.environ["RAYGUN_FASTA"]
OUT = pathlib.Path(os.environ.get("RAYGUN_OUT", "/tmp/raygun_results"))
OUT.mkdir(parents=True, exist_ok=True)
SKIP_PERF = os.environ.get("RAYGUN_SKIP_PERF", "") == "1"


def read_fasta(p):
    recs, h, s = [], None, []
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(s)))
            h, s = line[1:], []
        elif line:
            s.append(line)
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


def peak_mem():
    if DEV == "sdaa":
        return torch.sdaa.max_memory_allocated() / 2**30
    if DEV == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    return 0.0


def sync():
    if DEV == "sdaa":
        torch.sdaa.synchronize()
    elif DEV == "cuda":
        torch.cuda.synchronize()


def to_np(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    return x.detach().float().cpu().numpy()


def main():
    torch.manual_seed(42)
    from esm.pretrained import esm2_t33_650M_UR50D
    from raygun.modelv2.raygun import Raygun as RaygunV2
    from raygun.modelv2.ltraygun import RaygunLightning

    device = torch.device(DEV)
    t0 = time.time()
    ckpt = torch.load(os.environ["RAYGUN_CKPT"], map_location="cpu", weights_only=False)
    esmmodel, esmalphabet = esm2_t33_650M_UR50D()
    rmod = RaygunV2(numencoders=12, numdecoders=12, fixed_esm_batching=False)
    rayl = RaygunLightning(rmod, esmmodel)
    rayl.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt, rayl
    esmmodel = esmmodel.to(device).eval()
    raymodel = rmod.to(device).eval()
    bc = esmalphabet.get_batch_converter()
    print("[%s] weight load %.1fs" % (DEV, time.time() - t0), flush=True)

    recs = read_fasta(FASTA)
    print("[%s] templates: %d | skip_perf=%s" % (DEV, len(recs), SKIP_PERF), flush=True)

    results, logits_store = [], {}
    peak = 0.0
    with torch.no_grad():
        for i, (h, seq) in enumerate(recs):
            glen = max(32, int(round(0.75 * len(seq))))
            entry = {"name": h, "seqlen": len(seq), "genlen": glen}
            _, _, tokens = bc([("prot", seq)])
            tokens = tokens.to(device)
            emb = esmmodel(tokens, repr_layers=[33], return_contacts=False)[
                "representations"][33][:, 1:-1, :]
            if not SKIP_PERF:
                times = []
                for r in range(4):  # 1 warmup + 3 timed
                    sync(); t = time.time()
                    mem = raymodel.encoder(emb, noise=1.0)
                    out = raymodel.decoder(mem, glen)
                    _ = raymodel.esmdecoder(out)
                    sync()
                    if r > 0:
                        times.append(round(time.time() - t, 4))
                    peak = max(peak, peak_mem())
                entry["times"] = times
                entry["time_med"] = float(np.median(times))
            # 数值臂: noise=0 确定性前向
            mem0 = raymodel.encoder(emb, noise=0.0)
            out0 = raymodel.decoder(mem0, glen)
            logits_store[h] = to_np(raymodel.esmdecoder(out0)).astype(np.float32)
            peak = max(peak, peak_mem())
            results.append(entry)
            msg = "%3d/%d %-14s len=%4d gen=%4d" % (i + 1, len(recs), h[:14], len(seq), glen)
            if "time_med" in entry:
                msg += " t=%.3fs" % entry["time_med"]
            print(msg, flush=True)

    meds = [e["time_med"] for e in results if "time_med" in e]
    summary = {"dev": DEV, "n": len(recs), "peak_gib": round(peak, 2)}
    if meds:
        summary.update({"median_sec": round(float(np.median(meds)), 4),
                        "mean_sec": round(float(np.mean(meds)), 4),
                        "p90_sec": round(float(np.percentile(meds, 90)), 4),
                        "total_sec": round(float(np.sum(meds)), 2)})
    json.dump(summary, open(OUT / ("batch_%s.json" % DEV), "w"), indent=1)
    json.dump({"per_seq": results}, open(OUT / ("batch_%s_perseq.json" % DEV), "w"))
    np.savez(str(OUT / ("batch_%s_logits.npz" % DEV)), **logits_store)
    print("[done]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
