# -*- coding: utf-8 -*-
"""批量 logits 比对 v2：noise=0 确定性生成场景的精度指标。

logits (genlen,32) → softmax = 逐位置氨基酸采样分布；argmax = 实际生成序列。
指标（每条）:
  top1/top3/top5 agreement —— 逐位置 top-k 氨基酸一致率（top1=1.0 即生成序列逐位一致）
  js  —— 逐位置 JS 散度均值（分布层面差异，bits：log2）
  cos / mean_rel
用法: python3 compare_batch.py <a.npz> <b.npz> [out.json]
"""
import sys, json
import numpy as np

A, B = np.load(sys.argv[1]), np.load(sys.argv[2])


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


rows = []
for k in sorted(A.files):
    if k not in B.files:
        print("missing in B:", k)
        continue
    a = A[k].astype(np.float64)
    b = B[k].astype(np.float64)
    if a.shape != b.shape:
        print("shape mismatch:", k, a.shape, b.shape)
        continue
    d = np.abs(a.ravel() - b.ravel())
    den = np.clip(np.maximum(np.abs(a.ravel()), np.abs(b.ravel())), 1e-12, None)
    pa = softmax(a)
    pb = softmax(b)
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * (np.log2(np.clip(p, 1e-12, None)) - np.log2(np.clip(q, 1e-12, None))), axis=-1)
    js = (0.5 * kl(pa, m) + 0.5 * kl(pb, m)).mean()
    ta, tb = np.argsort(-a, axis=-1), np.argsort(-b, axis=-1)
    rows.append({
        "name": k, "genlen": int(a.shape[0]),
        "top1": float((ta[..., 0] == tb[..., 0]).mean()),
        "top3": float(np.mean([t in list(tb[i, :3, 0]) for i, t in enumerate(ta[:, 0, 0])])),
        "top5": float(np.mean([t in list(tb[i, :5, 0]) for i, t in enumerate(ta[:, 0, 0])])),
        "js": float(js),
        "cos": float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))),
        "mean_rel": float((d / den).mean()),
        "max_abs": float(d.max()),
    })

t1 = [r["top1"] for r in rows]
summary = {
    "n": len(rows),
    "seq_identical": int(sum(1 for x in t1 if x == 1.0)),
    "top1_min": float(np.min(t1)),
    "top1_mean": float(np.mean(t1)),
    "top3_mean": float(np.mean([r["top3"] for r in rows])),
    "top5_mean": float(np.mean([r["top5"] for r in rows])),
    "js_median": float(np.median([r["js"] for r in rows])),
    "js_max": float(np.max([r["js"] for r in rows])),
    "cos_min": float(np.min([r["cos"] for r in rows])),
    "mean_rel_max": float(np.max([r["mean_rel"] for r in rows])),
}
print("SUMMARY:", json.dumps(summary, indent=1))
worst = sorted(rows, key=lambda r: -r["js"])[:5]
print("WORST5_by_js:", json.dumps([{k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in worst], indent=1))
if len(sys.argv) > 3:
    json.dump({"summary": summary, "per_seq": rows}, open(sys.argv[3], "w"), indent=1)
