from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "dis2_port"))
from common import DEV, load_cg0_teacher, make_student  # noqa: E402
from dis2_port.train_dis2 import (  # noqa: E402
    MISS_NDSM, MISS_RGB, DLKD_ver4, model_input, official_loss,
)
from hinton_kd.train_hinton import soft_kd  # noqa: E402
from ours_s2_2stage.train_s2_2stage import kd_terms  # noqa: E402
from models_bank import criterion_rs  # noqa: E402


def main():
    torch.manual_seed(42)
    result = {"device": str(DEV), "checks": {}}
    teacher = load_cg0_teacher()
    student = make_student()
    x, d, y = torch.randn(1, 5, 128, 128, device=DEV), torch.randn(1, 5, 128, 128, device=DEV), \
              torch.randint(0, 2, (1, 128, 128), device=DEV).float()
    with torch.no_grad():
        ta, td = teacher.forward_pair(x, d)
        sout = student(x)
        result["checks"]["teacher_student_shapes"] = [list(ta["z"].shape), list(sout["z"].shape)]
        result["checks"]["hinton_soft_kd_finite"] = bool(torch.isfinite(soft_kd(sout["z"][:, 0], ta["z"][:, 0])).item())
    student.train()
    lp, le, oa, od = kd_terms(student, teacher, x, d)
    s2loss = lp + le + (oa["z"].mean() * 0)
    s2loss.backward()
    result["checks"]["s2_policy_evidence_backward_finite"] = bool(torch.isfinite(s2loss).item())
    result["checks"]["s2_loss_values"] = [float(lp.detach()), float(le.detach())]
    del student, teacher, ta, td, sout, oa, od
    if DEV.type == "cuda":
        torch.cuda.empty_cache()

    dis2 = DLKD_ver4(num_classes=2, image_size=(128, 128),
                     ndsm_encoder_channels=5, rgb_encoder_channels=5).to(DEV)
    dis2.is_training = True
    labels = criterion_rs.expand_target(y.long(), n_class=2).float()
    directional = []
    for a, b, miss in ((x, d, MISS_NDSM.to(DEV)), (d, x, MISS_RGB.to(DEV))):
        with torch.autocast(device_type="cuda", enabled=(DEV.type == "cuda")):
            outputs = dis2(model_input(a, b), [miss], device=DEV)
            loss, terms = official_loss(outputs, labels, epoch=6)
        loss.backward()
        directional.append(float(loss.detach().float().item()))
        del outputs, terms, loss
        dis2.zero_grad(set_to_none=True)
    result["checks"]["dis2_both_training_orientations_backward"] = directional
    dis2.eval()
    dis2.is_training = False
    with torch.no_grad():
        pa = dis2(model_input(x, torch.zeros_like(d)), [MISS_NDSM.to(DEV)], device=DEV)["seg_pred"]
        pd = dis2(model_input(torch.zeros_like(x), d), [MISS_RGB.to(DEV)], device=DEV)["seg_pred"]
    result["checks"]["dis2_target_only_shapes"] = [list(pa.shape), list(pd.shape)]
    result["checks"]["dis2_probabilities_finite"] = bool(torch.isfinite(pa).all() and torch.isfinite(pd).all())
    path = HERE.parent / "smoke_report.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
