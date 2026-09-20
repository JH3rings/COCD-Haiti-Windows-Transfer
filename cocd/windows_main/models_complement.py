"""New COCD -- deformable complement extraction (teacher) and orbit-conditioned
complement prediction (student).

This module only *adds* the two networks of the frozen design; it touches
nothing that already exists.  ``models.landslide_cocd``,
``models.landslide_cocd_v2``, ``losses.*`` and every shipped script are left
exactly as they were, so the legacy arms keep their original definitions.

The design, written out once:

    teacher   F_t, F_c            <- shared ConvNeXt-Tiny + FPN, fused level p2
              H_tc  = DCA(F_t, F_c)
              C_T   = Phi(F_t, H_tc) - Phi(F_t, DCA(F_t, 0))
              F_dual = F_t + C_T
              z_self = Head(F_t)        z_dual = Head(F_dual)
              L_T   = mean(BCE(z_self, y), BCE(z_dual, y))

    student   F_t   <- the same backbone, same fused level
              F_mod = (1 + gamma) * F_t + beta        (gamma, beta from OrbitID)
              C_S   = P(F_mod)
              F_S   = F_t + C_S
              z_S   = Head(F_S)

``Head`` is literally ``backbone.head`` followed by the same bilinear upsample
the single-orbit baseline uses, so N0's prediction path is reproduced bitwise
through the new interface.  ``C_T(F_c = 0) == 0`` is an algebraic identity of the
difference, not a constraint placed on ``DCA``: with the counter feature zero,
both terms are the same expression.

Injection is at the fused feature only, once, for both networks.  Nothing here
is multi-level, gated, or geometry-aware.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.landslide_cocd import ConvNeXtTinyFPN

__all__ = [
    'FUSED_CHANNELS', 'fused_p2', 'predict_from_fused', 'DCA', 'Phi',
    'NewCOCDTeacher', 'ComplementStudent',
    'ChannelLayerNorm', 'EARReaderWriter', 'EARTeacher', 'EARStudent',
    'init_ear_student_from_teacher',
]

FUSED_CHANNELS = 128


# --------------------------------------------------------------------------- #
# Evidence Allocation Reallocation (EAR)
# --------------------------------------------------------------------------- #
class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each spatial location, without affine terms."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm's normalized dimension is last, while FPN tensors are NCHW.
        return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],),
                            weight=None, bias=None, eps=1e-5).permute(0, 3, 1, 2)


class EARReaderWriter(nn.Module):
    """One shared 3x3 evidence reader/writer used by Teacher and Student.

    The only quantity supplied by the counter orbit is the low-dimensional
    guidance ``b``.  Keys and values are always extracted from the target
    feature passed as ``feat``.  The fixed nine offsets and the valid-neighbour
    mask make padding an unavailable candidate rather than a learnable value.
    """

    OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))

    def __init__(self, channels: int = FUSED_CHANNELS, dim: int = 32):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.ln = ChannelLayerNorm()
        self.wq = nn.Conv2d(channels, dim, 1, bias=False)
        self.wk = nn.Conv2d(channels, dim, 1, bias=False)
        self.wv = nn.Conv2d(channels, dim, 1, bias=False)
        self.wo = nn.Conv2d(dim, channels, 1, bias=False)

    @classmethod
    def _valid_mask(cls, h: int, w: int, device: torch.device) -> torch.Tensor:
        yy = torch.arange(h, device=device).view(h, 1).expand(h, w)
        xx = torch.arange(w, device=device).view(1, w).expand(h, w)
        masks = [((yy + dy >= 0) & (yy + dy < h) &
                  (xx + dx >= 0) & (xx + dx < w))
                 for dy, dx in cls.OFFSETS]
        return torch.stack(masks, dim=0).unsqueeze(0)  # (1, 9, H, W)

    def _project_candidates(self, feat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, _, h, w = feat.shape
        u = self.ln(feat)
        q0 = self.wq(u)
        # Unfold is ordered top-left to bottom-right, the same fixed order as
        # OFFSETS.  The spatial padding is removed from softmax by the mask.
        uk = F.unfold(u, kernel_size=3, padding=1)
        vf = F.unfold(feat, kernel_size=3, padding=1)
        k = self.wk(uk.view(b, self.channels, 9, h, w)
                    .permute(0, 2, 1, 3, 4).reshape(b * 9, self.channels, h, w))
        v = self.wv(vf.view(b, self.channels, 9, h, w)
                    .permute(0, 2, 1, 3, 4).reshape(b * 9, self.channels, h, w))
        k = k.view(b, 9, self.dim, h, w)
        v = v.view(b, 9, self.dim, h, w)
        return q0, k, v

    def forward(self, feat: torch.Tensor, guidance: torch.Tensor | None) -> dict:
        b, _, h, w = feat.shape
        q0, k, v = self._project_candidates(feat)
        valid = self._valid_mask(h, w, feat.device)
        scale = math.sqrt(self.dim)
        scores0 = (q0[:, None] * k).sum(dim=2) / scale
        scores0 = scores0.masked_fill(~valid, torch.finfo(scores0.dtype).min)
        a0 = torch.softmax(scores0, dim=1)
        if guidance is None:
            a = a0
            rho = torch.zeros_like(a0)
            correction = torch.zeros_like(feat)
        else:
            if guidance.shape != (b, self.dim, h, w):
                raise ValueError(f'EAR guidance shape {tuple(guidance.shape)} does not match '
                                 f'({b}, {self.dim}, {h}, {w})')
            scores = ((q0 + guidance)[:, None] * k).sum(dim=2) / scale
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            a = torch.softmax(scores, dim=1)
            rho = a - a0
            correction = self.wo((rho[:, :, None] * v).sum(dim=1))
        return {'a0': a0, 'a': a, 'rho': rho, 'C': correction}


class EARTeacher(nn.Module):
    """Counter-guided target-orbit Teacher for the evidence-reallocation study."""

    def __init__(self, channels: int = FUSED_CHANNELS, dim: int = 32,
                 n_points: int = 4):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.dca = DCA(channels, n_points=n_points)
        self.A = nn.Conv2d(channels, dim, 1, bias=False)
        nn.init.zeros_(self.A.weight)
        self.ear = EARReaderWriter(channels, dim)

    def encode_fused(self, x: torch.Tensor) -> torch.Tensor:
        return fused_p2(self.backbone, x)

    def predict(self, p2: torch.Tensor) -> torch.Tensor:
        return predict_from_fused(self.backbone.head, p2)

    def from_fused(self, target_fused: torch.Tensor,
                   counter_fused: torch.Tensor) -> dict:
        # The target feature owns every EAR key/value.  DCA is the only reader
        # of the counter feature and its output is reduced to bT by A.
        h = self.dca(target_fused, counter_fused)
        b_t = self.A(self.ear.ln(h))
        self_read = self.ear(target_fused, None)
        dual_read = self.ear(target_fused, b_t)
        return {
            'z_self': self.predict(target_fused),
            'z_dual': self.predict(target_fused + dual_read['C']),
            'F_t': target_fused,
            'C_T': dual_read['C'],
            'bT': b_t,
            'a0': dual_read['a0'],
            'a': dual_read['a'],
            'rho': dual_read['rho'],
            'self_rho': self_read['rho'],
        }

    def forward(self, target: torch.Tensor, counter: torch.Tensor) -> dict:
        return self.from_fused(self.encode_fused(target), self.encode_fused(counter))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor) -> tuple[dict, dict]:
        # Encode the two orbit views in one batched call.  This is still one
        # encoding per orbit, while keeping the Teacher-self and copied
        # Student-self paths numerically aligned on the same batch.
        fused = self.encode_fused(torch.cat((asc, desc), dim=0))
        f_asc, f_desc = fused.chunk(2, dim=0)
        return self.from_fused(f_asc, f_desc), self.from_fused(f_desc, f_asc)


class EARStudent(nn.Module):
    """Single-orbit deployment Student that predicts only the EAR guidance."""

    def __init__(self, channels: int = FUSED_CHANNELS, dim: int = 32):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.ear = EARReaderWriter(channels, dim)
        self.driver = nn.Sequential(
            nn.Conv2d(channels + 1, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        nn.init.zeros_(self.driver[-1].weight)
        nn.init.zeros_(self.driver[-1].bias)

    def predict(self, p2: torch.Tensor) -> torch.Tensor:
        return predict_from_fused(self.backbone.head, p2)

    def forward(self, x: torch.Tensor) -> dict:
        f_s = fused_p2(self.backbone, x)
        orbit = F.interpolate(x[:, 4:5], size=f_s.shape[-2:], mode='nearest')
        b_s = self.driver(torch.cat((self.ear.ln(f_s), orbit), dim=1))
        read = self.ear(f_s, b_s)
        z_self = self.predict(f_s)
        z = self.predict(f_s + read['C'])
        return {
            'z': z,
            'z_self': z_self,
            'F_t': f_s,
            'C_S': read['C'],
            'bS': b_s,
            'a0': read['a0'],
            'a': read['a'],
            'rho': read['rho'],
        }


def init_ear_student_from_teacher(student: EARStudent, teacher: EARTeacher) -> None:
    """Copy the trained Teacher's target path; leave the zero driver untouched."""
    student.backbone.load_state_dict(teacher.backbone.state_dict())
    student.ear.load_state_dict(teacher.ear.state_dict())


# --------------------------------------------------------------------------- #
# The shared trunk, exposed as a first-class fused feature
# --------------------------------------------------------------------------- #
def fused_p2(backbone: ConvNeXtTinyFPN, x: torch.Tensor) -> torch.Tensor:
    """The fused semantic feature ``p2``: 128 channels at 1/4 resolution.

    For a 128x128 input this is 32x32.  The arithmetic is the same expression
    ``ConvNeXtTinyFPN.decode`` evaluates internally before it applies the head,
    written out so the tensor can be held, added to and handed to a head
    separately.  Everything below the head is unchanged.
    """
    c2, c3, c4, c5 = backbone.encode(x)
    p5 = backbone.l5(c5)
    p4 = backbone.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
    p3 = backbone.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
    return backbone.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')


def predict_from_fused(head: nn.Module, p2: torch.Tensor) -> torch.Tensor:
    """Apply the shared prediction head to a fused feature and upsample to 128.

    ``head`` is ``backbone.head``, so every network in this file predicts through
    the identical classifier the single-orbit baseline uses.
    """
    return F.interpolate(head(p2), size=(128, 128), mode='bilinear', align_corners=False)


# --------------------------------------------------------------------------- #
# The only new module: deformable cross-orbit attention
# --------------------------------------------------------------------------- #
class DCA(nn.Module):
    """Deformable cross-orbit attention.

    For every position of the target feature ``F_t`` this samples ``n_points``
    learned offsets from the counter feature ``F_c`` and weights them by a softmax
    over query-key similarity.  It exists because ascending and descending
    geometry is not pixel-to-pixel aligned: a query has to look in a
    neighbourhood of the other orbit rather than at the same coordinate.

    Two properties are deliberate and are asserted in the unit tests rather than
    assumed:

    * with ``F_c = 0`` the output is exactly zero.  The value path has no bias
      and the output projection has no bias, so a weighted sum of zero samples is
      zero.  This is what makes the zero-counter reference of ``C_T`` numerically
      exact as well as algebraically exact.
    * the offsets are bounded, so a randomly initialised module cannot sample far
      outside the feature map on the first step.
    """

    def __init__(self, channels: int = FUSED_CHANNELS, n_points: int = 4,
                 hidden: int = 64, offset_bound: float = 3.0):
        super().__init__()
        self.n_points = n_points
        self.offset_bound = offset_bound
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.off = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2 * n_points, 1),
        )
        self.out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, f_t: torch.Tensor, f_c: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f_t.shape
        # (b, 2K, h, w) -> offsets in pixels, bounded to +/- offset_bound
        offsets = self.offset_bound * torch.tanh(self.off(f_t))
        offsets = offsets.view(b, self.n_points, 2, h, w)

        gy, gx = torch.meshgrid(
            torch.arange(h, device=f_t.device, dtype=f_t.dtype),
            torch.arange(w, device=f_t.device, dtype=f_t.dtype),
            indexing='ij')

        sampled = []
        for k in range(self.n_points):
            dx = offsets[:, k, 0]                      # (b, h, w) in pixels
            dy = offsets[:, k, 1]
            # pixel index i + dy maps to align_corners=False grid coord
            # (2 * (i + dy) + 1) / size - 1
            grid = torch.stack(((2.0 * (gx + dx) + 1.0) / w - 1.0,
                                (2.0 * (gy + dy) + 1.0) / h - 1.0), dim=-1)
            sampled.append(F.grid_sample(f_c, grid, mode='bilinear',
                                         padding_mode='zeros', align_corners=False))
        sampled = torch.stack(sampled, dim=1)          # (b, K, C, h, w)

        # similarity of the target query against each sampled counter value
        score = (self.q(f_t)[:, None] * sampled).sum(2) / math.sqrt(c)
        attn = torch.softmax(score, dim=1)             # (b, K, h, w)
        return self.out((attn[:, :, None] * sampled).sum(1))


class Phi(nn.Module):
    """Plain convolutional projection of ``(F_t, H_tc)``.

    The last layer starts at zero so a freshly built teacher is exactly its
    single-orbit self and grows the complement from nothing.  That is the same
    convention the existing ``Correction`` and ``Driver`` blocks use, so every
    arm in the project starts from the same uncorrected function.
    """

    def __init__(self, channels: int = FUSED_CHANNELS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((f_t, h), 1))


# --------------------------------------------------------------------------- #
# Teacher
# --------------------------------------------------------------------------- #
class NewCOCDTeacher(nn.Module):
    """Shared target/counter encoder and a difference-defined complement.

    One module is added to the single-orbit network: ``DCA`` together with the
    plain projection ``Phi`` that turns its output into the complement.  The
    complement is injected once, at the fused feature, and both readings come out
    of the same head.
    """

    def __init__(self, channels: int = FUSED_CHANNELS, n_points: int = 4):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.dca = DCA(channels, n_points=n_points)
        self.phi = Phi(channels)

    # -- trunk ------------------------------------------------------------- #
    def encode_fused(self, x: torch.Tensor) -> torch.Tensor:
        """One encoder call, one fused feature.  Encoder weights are shared."""
        return fused_p2(self.backbone, x)

    def predict(self, p2: torch.Tensor) -> torch.Tensor:
        """Both readings go through the identical head."""
        return predict_from_fused(self.backbone.head, p2)

    # -- complement -------------------------------------------------------- #
    def complement(self, p2_t: torch.Tensor, p2_c: torch.Tensor) -> torch.Tensor:
        """``C_T = Phi(F_t, DCA(F_t, F_c)) - Phi(F_t, DCA(F_t, 0))``.

        The second term is the same expression with the counter feature set to
        zero, so ``C_T`` is exactly zero when ``F_c`` is zero -- by construction,
        with no constraint on ``DCA``.  The return value is the marginal
        contribution of the counter orbit, not a fused feature.
        """
        h = self.dca(p2_t, p2_c)
        h_zero = self.dca(p2_t, torch.zeros_like(p2_c))
        return self.phi(p2_t, h) - self.phi(p2_t, h_zero)

    def from_fused(self, p2_t: torch.Tensor, p2_c: torch.Tensor) -> dict:
        """Both readings from already-encoded features.

        Kept separate from ``forward_pair`` so the audit can hold one target
        feature fixed and vary only the counter, reusing the target encoding.
        """
        c_t = self.complement(p2_t, p2_c)
        f_dual = p2_t + c_t
        return {'z_self': self.predict(p2_t), 'z_dual': self.predict(f_dual),
                'F_t': p2_t, 'F_dual': f_dual, 'C_T': c_t}

    # -- deployment-shaped entry points ------------------------------------ #
    def forward(self, target: torch.Tensor, counter: torch.Tensor) -> dict:
        return self.from_fused(self.encode_fused(target), self.encode_fused(counter))

    def forward_pair(self, asc: torch.Tensor, desc: torch.Tensor
                     ) -> tuple[dict, dict]:
        """Both target directions with exactly two encoder calls, not four."""
        p2_a, p2_d = self.encode_fused(asc), self.encode_fused(desc)
        return self.from_fused(p2_a, p2_d), self.from_fused(p2_d, p2_a)


# --------------------------------------------------------------------------- #
# Student
# --------------------------------------------------------------------------- #
class ComplementStudent(nn.Module):
    """Single-orbit deployment network with an optional complement branch.

    ``complement=False`` is N0: the single-orbit baseline, fused feature into the
    shared head, nothing else.  ``complement=True`` adds the orbit-conditioned
    complement predictor and one injection point.

    ``orbit_embedding=False`` keeps the predictor but drops the FiLM
    conditioning, which is the one difference between N2 and N2 w/o Orbit
    Embedding.  That arm exists to answer whether the orbit condition carries
    anything, so it may not differ from N2 in any other respect.

    Neither setting touches the segmentation path: the head always reads the
    unmodulated ``F_t + C_S`` when there is a complement, and ``F_t`` alone when
    there is not, so N0 stays a clean single-orbit baseline.
    """

    def __init__(self, complement: bool = True, orbit_embedding: bool = True,
                 channels: int = FUSED_CHANNELS, hidden: int = 64,
                 orbit_dim: int = 32):
        super().__init__()
        self.backbone = ConvNeXtTinyFPN()
        self.complement_on = complement
        self.orbit_on = bool(orbit_embedding and complement)
        self.channels = channels
        if complement:
            if self.orbit_on:
                self.orbit_emb = nn.Embedding(2, orbit_dim)
                self.orbit_mlp = nn.Sequential(
                    nn.Linear(orbit_dim, hidden), nn.GELU(),
                    nn.Linear(hidden, 2 * channels),
                )
                # gamma = beta = 0 at initialisation, so the modulation starts as
                # the identity and the orbit condition has to earn its effect.
                nn.init.zeros_(self.orbit_mlp[-1].weight)
                nn.init.zeros_(self.orbit_mlp[-1].bias)
            self.predictor = nn.Sequential(
                nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, channels, 1),
            )
            nn.init.zeros_(self.predictor[-1].weight)
            nn.init.zeros_(self.predictor[-1].bias)

    def predict(self, p2: torch.Tensor) -> torch.Tensor:
        return predict_from_fused(self.backbone.head, p2)

    def modulate(self, f_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``(1 + gamma) * F_t + beta``, with the orbit read from channel 4.

        The orbit plane is a spatially constant channel already present in the
        input, so no new data is needed.  The modulation feeds the complement
        branch only.
        """
        orbit = (x[:, 4].mean(dim=(1, 2)) > 0.5).long()
        gb = self.orbit_mlp(self.orbit_emb(orbit))
        gamma = gb[:, :self.channels, None, None]
        beta = gb[:, self.channels:, None, None]
        return (1.0 + gamma) * f_t + beta

    def forward(self, x: torch.Tensor) -> dict:
        p2 = fused_p2(self.backbone, x)
        if not self.complement_on:
            return {'z': self.predict(p2), 'F_t': p2, 'F_mod': p2, 'C_S': None}
        f_mod = self.modulate(p2, x) if self.orbit_on else p2
        c_s = self.predictor(f_mod)
        return {'z': self.predict(p2 + c_s), 'F_t': p2, 'F_mod': f_mod, 'C_S': c_s}


# --------------------------------------------------------------------------- #
# Model definitions of the active plan
# --------------------------------------------------------------------------- #
# ``kd`` mirrors the legacy config field so the runner can treat both families
# through one shape.  Nothing here overrides a legacy definition; these are the
# new method's arms.
COMPLEMENT_CONFIGS = {
    # single-orbit baseline: fused feature -> shared head, no complement branch
    'N0': {'complement': False, 'orbit_embedding': False, 'distil': None,
           'label': 'Single-Orbit'},
    # complement predictor trained with the segmentation loss only
    'N1': {'complement': True, 'orbit_embedding': True, 'distil': None,
           'label': 'Complement-Predictor'},
    # full COCD: N1 plus the complement distillation term
    'N2': {'complement': True, 'orbit_embedding': True, 'distil': 'complement',
           'label': 'COCD-Full'},
    # full COCD without the orbit conditioning, the one internal ablation
    'N2noorbit': {'complement': True, 'orbit_embedding': False, 'distil': 'complement',
                  'label': 'COCD-no-OrbitEmb'},
}


def build(model_kind: str) -> nn.Module:
    """Construct one of the two new networks by name."""
    if model_kind == 'teacher':
        return NewCOCDTeacher()
    if model_kind in COMPLEMENT_CONFIGS:
        cfg = COMPLEMENT_CONFIGS[model_kind]
        return ComplementStudent(complement=cfg['complement'],
                                 orbit_embedding=cfg['orbit_embedding'])
    raise ValueError(f'unknown model kind {model_kind!r}')


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
