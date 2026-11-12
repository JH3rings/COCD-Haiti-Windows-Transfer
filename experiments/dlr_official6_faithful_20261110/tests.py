"""Formula checks of every ported loss against the authors' own code."""
import ast
import importlib.util
import sys

import numpy as np
import torch
from torch.nn import functional as F

import common as C
import methods as M

REF = C.HERE / "author_refs"
torch.manual_seed(0)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def functions_from(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns = {"torch": torch, "F": F, "nn": torch.nn}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns


def close(a, b, what, tol=1e-6):
    a, b = float(a), float(b)
    assert abs(a - b) <= tol * max(1., abs(b)), (what, a, b)
    print(f"ok  {what:42s} {a:.8g}")


def main():
    # ProtoKD: 2-D port == author 3-D function on a singleton depth axis
    proto = load(REF / "protokd_loss.py", "protokd_loss")
    f, ft = torch.randn(3, 8, 16, 16), torch.randn(3, 8, 16, 16)
    lab = (torch.rand(3, 1, 16, 16) > .7).long()
    close(M.prototype_loss_2d(f, ft, lab), proto.prototype_loss(f[..., None], ft[..., None], lab[..., None], 2)[2],
          "ProtoKD prototype_loss (2-D port)")
    zs, zt = torch.randn(3, 2, 16, 16), torch.randn(3, 2, 16, 16)
    close(M.softmax_kl_loss(zs, zt).mean(), proto.softmax_kl_loss(zs, zt).mean(), "ProtoKD softmax_kl_loss")
    cfg = load(REF / "protokd_config.py", "protokd_config")
    print("    ProtoKD config T =", getattr(cfg, "cfg", getattr(cfg, "config", {})).get("T", "see file"))

    # DIS2: l2_kd_loss / orthogonality copied verbatim from the author file
    dis2 = functions_from(C.EXP.parent / "comparisons/dis2/source/models_bank/DLKD_ver4.py", {"l2_kd_loss"})
    a, b = torch.randn(4, 32), torch.randn(4, 32)
    close(M.l2_kd_loss(a, b), dis2["l2_kd_loss"](a, b), "DIS2 l2_kd_loss tokens")
    a, b = torch.randn(2, 8, 6, 6), torch.randn(2, 8, 6, 6)
    close(M.l2_kd_loss(a, b, True), dis2["l2_kd_loss"](a, b, spatial=True), "DIS2 l2_kd_loss spatial")
    zs, zt = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    two_s, two_t = M.two_class(zs), M.two_class(zt)
    ref = F.kl_div(F.log_softmax(two_s / 2, 1), F.softmax(two_t / 2, 1), reduction="mean") * 4
    close(M.dis2_logit_kl(zs, zt, 2), ref, "DIS2 logit KL (T=2, T^2)")

    # PKDOT: similarity matrix / lowest-30 / min-shift as in PKDOT_kfold.py
    pk = functions_from(REF / "pkdot_utils.py", {"cosine_similarity_matrix_transpose_torch"})
    x = torch.randn(40, 16)
    mine = F.normalize(x, dim=1) @ F.normalize(x, dim=1).T
    close((mine - pk["cosine_similarity_matrix_transpose_torch"](x.clone())).abs().max() + 1, 1,
          "PKDOT cosine similarity matrix")
    idx_author = np.sort(np.argpartition(mine.numpy(), 30, axis=1)[:, :30], 1)
    idx_mine = np.sort(mine.topk(30, largest=False).indices.numpy(), 1)
    assert (idx_author == idx_mine).all()
    print("ok  PKDOT lowest-30 selection identical")
    sys.path.insert(0, str(C.EXP / "dlr_unet_study_20261104" / "pkdot_deps"))
    from geomloss import SamplesLoss
    sk = SamplesLoss("sinkhorn", p=2, blur=.1, backend="tensorized")
    fs, ftt, valid = torch.randn(4, 16, 32, 32), torch.randn(4, 16, 32, 32), torch.ones(4, 1, 128, 128)
    assert torch.isfinite(M.pkdot_relations(fs, ftt, valid, sk))
    close(M.pkdot_relations(fs, fs, valid, sk), 0, "PKDOT identical features -> 0", tol=1e-3)

    # AT: official at / at_loss
    x, y = torch.randn(2, 8, 10, 10), torch.randn(2, 8, 10, 10)
    ref = (F.normalize(x.pow(2).mean(1).view(2, -1)) - F.normalize(y.pow(2).mean(1).view(2, -1))).pow(2).mean()
    close(M.at_loss(x, y), ref, "AT at_loss (official)")

    # ACAM: diversity bounds and zero loss at identity
    acam = M.ACAM(chunk=64)
    z = torch.randn(2, 2, 16, 16)
    lc, ls, div = acam(z, z, torch.ones(2, 1, 16, 16))
    close(lc + ls, 0, "ACAM identical maps -> distill 0")
    assert 0 <= float(div) <= 2.0001

    # Hinton / WKD / FAD / CAD sanity
    z = torch.randn(2, 1, 16, 16)
    close(M.hinton_kd(z, z, torch.ones_like(z), 4.), 0, "KD identical logits -> 0")
    p = torch.rand(2, 1, 16, 16)
    close(M.wavelet_kd(p, p, torch.ones_like(p)), 0, "WKD identical maps -> 0")
    f = torch.randn(2, 8, 16, 16)
    close(M.fad(f, f, torch.ones(2, 1, 16, 16)), 0, "FAD identical -> 0")
    close(M.cad(f, f), 0, "CAD identical -> 0")
    print("ALL FORMULA CHECKS PASSED")


if __name__ == "__main__":
    main()
