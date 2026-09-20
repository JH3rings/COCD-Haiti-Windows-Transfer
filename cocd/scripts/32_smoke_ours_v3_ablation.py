"""Pre-training verification for the Ours-V3 structural ablation.

Runs on CPU so it never competes with a training run for the MPS device.  It
checks the invariants the design depends on rather than just "the shapes line
up": that every arm starts exactly on the Teacher self path, that the inherited
operator is exactly zero for a zero driver, that R1 still reproduces the v2
network bit-for-bit, and that the R2/R3 difference lives only in where the P3
driver is read from.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A))
from models.landslide_cocd_v2 import Driver, OursV2Student, OursV2Teacher, OursV3Student, omega  # noqa: E402

DEV = torch.device('cpu')
RESULTS = []


def check(label, cond, extra=''):
    RESULTS.append(bool(cond))
    print(f'  [{"PASS" if cond else "FAIL"}] {label}{("  " + extra) if extra else ""}')


def build(variant, teacher):
    model = OursV3Student(variant).to(DEV)
    model.load_teacher_self(teacher)
    model.eval()
    return model


teacher = OursV2Teacher().to(DEV)
teacher.load_state_dict(torch.load(A / 'experiments' / 'ours_v2' / 'teacher.pt', map_location=DEV, weights_only=True))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

torch.manual_seed(0)
x = torch.randn(1, 5, 128, 128, device=DEV)
with torch.no_grad():
    zt_self = teacher(x, torch.randn(1, 5, 128, 128, device=DEV))[0]
bare = sum(p.numel() for p in OursV3Student('R0').backbone.parameters())

print('== 参数量 ==')
for v in ('R0', 'R1', 'R2', 'R3'):
    m = build(v, teacher)
    n = sum(p.numel() for p in m.parameters())
    print(f'  {v}: total {n/1e6:.4f} M   correction-side {n - bare:,}')

print('\n== 初始化不变量：零修正时学生 == Teacher self path ==')
for v in ('R0', 'R1', 'R2', 'R3'):
    with torch.no_grad():
        z4, z34, (r3, r4) = build(v, teacher)(x)
    print(f'  -- {v} --')
    check('z34 == Teacher self', float((z34 - zt_self).abs().max()) == 0.0)
    check('z4  == Teacher self', float((z4 - zt_self).abs().max()) == 0.0)
    check('初始残差恒为 0', max(float(r3.abs().max()), float(r4.abs().max())) == 0.0)

print('\n== 算子不变量：Omega(a, 0) == 0 ==')
with torch.no_grad():
    a128 = torch.randn(2, 128, 16, 16, device=DEV)
    for name, op in (('teacher.a3', teacher.a3), ('teacher.a4', teacher.a4)):
        check(f'{name}: Omega(a,0) == 0', float(omega(op, a128, torch.zeros_like(a128)).abs().max()) == 0.0)

print('\n== R1 必须与 v2 Student 逐元素一致 ==')
v2 = OursV2Student().to(DEV); v2.load_teacher_self(teacher); v2.eval()
v3 = build('R1', teacher)
v3.q3.load_state_dict(v2.q3.state_dict()); v3.q4.load_state_dict(v2.q4.state_dict())
with torch.no_grad():
    o2, o3 = v2(x), v3(x)
for i, nm in enumerate(('z4', 'z34')):
    check(f'v2 vs R1: {nm} 一致', float((o2[i] - o3[i]).abs().max()) == 0.0)
check('输出均为 (z4, z34, residuals)', len(o2) == len(o3) == 3)

print('\n== R2 与 R3：驱动权重完全相同，只有 P3 驱动来源不同 ==')
torch.manual_seed(7)
m2 = build('R2', teacher)
with torch.no_grad():
    for d in (m2.d3, m2.d4):
        torch.nn.init.normal_(d.net[-1].weight, std=0.02)
        torch.nn.init.normal_(d.net[-1].bias, std=0.02)
m3 = OursV3Student('R3').to(DEV)
m3.load_state_dict(m2.state_dict())
m3.eval()
with torch.no_grad():
    z4_2, z34_2, _ = m2(x)
    z4_3, z34_3, _ = m3(x)
check('z34 不同（P3 驱动确实被递进改写）', float((z34_2 - z34_3).abs().max()) > 1e-6,
      f'max|d|={float((z34_2 - z34_3).abs().max()):.3e}')
check('z4 完全相同（z4 只经 r4，与 r3 无关）', float((z4_2 - z4_3).abs().max()) == 0.0)

print('\n== 形状与监督目标一致性 ==')
check('z4/z34 形状 (1,1,128,128)', tuple(z34_3.shape) == (1, 1, 128, 128), str(tuple(z34_3.shape)))
check('r3 形状 (1,128,16,16)', tuple(o3[2][0].shape) == (1, 128, 16, 16), str(tuple(o3[2][0].shape)))
check('r4 形状 (1,128,8,8)', tuple(o3[2][1].shape) == (1, 128, 8, 8), str(tuple(o3[2][1].shape)))

print('\n== Driver 结构（文档要求 3x3, 128->32->128, 末层零初始化）==')
dd = Driver(128)
ks = [(c.kernel_size, c.in_channels, c.out_channels) for c in dd.net if isinstance(c, torch.nn.Conv2d)]
check('kernel/通道与文档一致', ks == [((3, 3), 128, 32), ((3, 3), 32, 128)], str(ks))
check('末层零初始化', bool((dd.net[-1].weight == 0).all() and (dd.net[-1].bias == 0).all()))
check('单个 Driver = 73,888', sum(p.numel() for p in dd.parameters()) == 73888)

print(f'\n=== {sum(RESULTS)}/{len(RESULTS)} 断言通过 ===')
sys.exit(0 if all(RESULTS) else 1)
