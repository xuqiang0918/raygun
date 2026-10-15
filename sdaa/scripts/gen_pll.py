# -*- coding: utf-8 -*-
"""Raygun 生成质量评估（官方 get_PLL 置信度）。

用法:
    python3 gen_pll.py gen <dev>   # 对 RAYGUN_FASTA 每条模板 noise=1.0 生成 1 条 -> gen_<dev>.fasta
    python3 gen_pll.py pll <dev>   # 对 RAYGUN_IN fasta 算官方 PLL -> pll_<tag>.json
env: RAYGUN_CKPT / RAYGUN_FASTA / RAYGUN_OUT / RAYGUN_IN / RAYGUN_TAG
"""
import os, sys, json, pathlib
import numpy as np
import torch

mode, DEV = sys.argv[1], sys.argv[2]
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = pathlib.Path(os.environ["RAYGUN_OUT"])
OUT.mkdir(parents=True, exist_ok=True)


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


def main():
    torch.manual_seed(42)
    from esm.pretrained import esm2_t33_650M_UR50D
    from raygun.modelv2.raygun import Raygun as RaygunV2
    from raygun.modelv2.ltraygun import RaygunLightning

    device = torch.device(DEV)
    ckpt = torch.load(os.environ["RAYGUN_CKPT"], map_location="cpu", weights_only=False)
    esmmodel, alphabet = esm2_t33_650M_UR50D()
    rmod = RaygunV2(numencoders=12, numdecoders=12, fixed_esm_batching=False)
    rayl = RaygunLightning(rmod, esmmodel)
    rayl.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt, rayl
    esmmodel = esmmodel.to(device).eval()
    raymodel = rmod.to(device).eval()
    bc = alphabet.get_batch_converter()

    if mode == "gen":
        recs = read_fasta(os.environ["RAYGUN_FASTA"])
        fasta = OUT / ("gen_%s.fasta" % DEV)
        with torch.no_grad(), open(fasta, "w") as f:
            for i, (h, seq) in enumerate(recs):
                glen = max(32, int(round(0.75 * len(seq))))
                _, _, tokens = bc([("prot", seq)])
                tokens = tokens.to(device)
                emb = esmmodel(tokens, repr_layers=[33], return_contacts=False)[
                    "representations"][33][:, 1:-1, :]
                mem = raymodel.encoder(emb, noise=1.0)
                s = raymodel.get_sequences_from_fixed(mem, glen)[0]
                f.write(">%s dev=%s glen=%d\n%s\n" % (h, DEV, glen, s))
                if (i + 1) % 20 == 0:
                    print("%d/%d" % (i + 1, len(recs)), flush=True)
        print("[done] generated ->", fasta, flush=True)
    else:
        from raygun.pll import get_PLL
        tag = os.environ.get("RAYGUN_TAG", "x")
        recs = read_fasta(os.environ["RAYGUN_IN"])
        rows = []
        with torch.no_grad():
            for i, (h, seq) in enumerate(recs):
                pll = float(get_PLL(seq, esmmodel, alphabet, bc,
                                    reduce=np.sum, device=DEV))
                norm = pll / abs(-0.406 * len(seq) + 1.363)
                rows.append({"name": h, "len": len(seq), "pll": round(pll, 2),
                             "pll_norm": round(norm, 4)})
                if (i + 1) % 20 == 0:
                    print("%d/%d" % (i + 1, len(recs)), flush=True)
        npll = [r["pll_norm"] for r in rows]
        out = {"tag": tag, "n": len(rows),
               "pll_norm_median": round(float(np.median(npll)), 4),
               "pll_norm_mean": round(float(np.mean(npll)), 4),
               "pll_norm_p10": round(float(np.percentile(npll, 10)), 4),
               "pll_norm_p90": round(float(np.percentile(npll, 90)), 4),
               "per_seq": rows}
        json.dump(out, open(OUT / ("pll_%s.json" % tag), "w"), indent=1)
        print("SUMMARY:", json.dumps({k: v for k, v in out.items() if k != "per_seq"}), flush=True)


if __name__ == "__main__":
    main()
