"""Single-orbit baselines under the unified v4 protocol.

Four seats, all retrained from scratch under the SAME protocol the internal
arms use, so the main table is finally internally comparable:

    Boehm SAR U-Net++      smp.UnetPlusPlus  encoder resnet34
    CDNetE Early Fusion    smp.Unet          encoder resnet34
    MFEWF adapted          our DRN/AMM/CAASP/MFFRM reimplementation
    FC-Siam-diff           official SiamUnet_diff, adapted input

``resnet34`` is used for every encoder-based seat, so the comparison isolates
the method (decoder topology, fusion contract, method-specific loss) rather
than backbone capacity. The official Boehm notebook ships resnet50; that is a
documented deviation, disclosed in the result table.

What is shared with every other method (protocol v4, see
``cocd/configs/protocol_windows_main.json``):

    Adam 5e-5, wd 0, no scheduler, physical batch 16,
    patience-2 early stopping on test AUPRC, ceiling 200 epochs,
    the best-monitored-epoch checkpoint IS the model,
    binary threshold permanently fixed at 0.5, AUPRC threshold-free.

What is allowed to differ, per the protocol's fairness rule: the architecture,
and the loss only where it is inseparable from the published method.

--------------------------------------------------------------------------------
MONITOR RISK — read before quoting any number produced by this script
--------------------------------------------------------------------------------
The early-stopping rule reads TEST AUPRC. This departure from the v3 rule that
test never changes a model was chosen deliberately by the user on 2026-09-19
("仍然用 test 监控，但标注风险") and is recorded in ``monitor_risk`` in the
protocol file. Consequence: the reported test AUPRC is the maximum over the
epochs visited and carries an optimistic bias of unknown size. Methods stay
comparable to each other only because every method is selected under the
identical rule. The 0.5 threshold is threshold-free-relevant and is never used
to pick an epoch; no architecture, loss or lambda is chosen from test.
--------------------------------------------------------------------------------

Run:
    python cocd/scripts/23_train_external_baselines.py --seat boehm
    python cocd/scripts/23_train_external_baselines.py --seat all
"""

import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
COCD = ROOT / 'cocd'
A = COCD
sys.path.insert(0, str(COCD))

# ---------------------------------------------------------------------------
# shared final data path: external baselines use the same loader/split as the
# final-v2 method, without importing a retired self-owned model runner
# ---------------------------------------------------------------------------
from paths import DEV  # noqa: E402
from windows_main.data import HaitiPairs, positive_weight, split  # noqa: E402


def loader(dataset, shuffle=False):
    """The same DataLoader the internal arms use, with the protocol's batch."""
    return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle, num_workers=0)

PROT = json.loads((COCD / 'configs' / 'protocol_windows_main.json').read_text(encoding='utf-8'))
SEED = PROT['seeds']['main']
_SCHED = PROT['schedule']
MAX_EPOCHS = _SCHED['max_epochs']                 # 200, a ceiling, not a plan
PATIENCE = _SCHED['early_stop_patience']          # 2
EARLY_STOP = _SCHED['early_stop']                 # True
MONITOR = _SCHED['monitor']                       # 'test_auprc'
BATCH = PROT['batch']['physical']                 # 16
LR = PROT['optimizer']['lr']                      # 5e-5
WD = PROT['optimizer']['weight_decay']            # 0.0
THRESHOLD = 0.5                                   # fixed, permanent

OUT_ROOT = ROOT / 'experiments' / 'single_orbit_baselines_v3'
LOG_DIR = OUT_ROOT / 'logs'
THIRD = COCD / 'third_party' / 'landslide_baselines'
ENCODER = 'resnet34'                             # one trunk for every seat
EXPECTED_TRAIN_VIEWS = 8220                      # 1370 locations x 3 modes x 2 orbits

# the disclosure below is written into every log and every progress file
MONITOR_RISK = ('EARLY STOPPING MONITORS THE TEST PARTITION: the reported test '
                'AUPRC is the best-over-epochs value and is optimistically biased. '
                'Threshold stays fixed at 0.5; every method uses the identical rule.')


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def metric(p, y, t=THRESHOLD):
    """Shared fixed-threshold summary used by the external comparison seats."""
    p = np.asarray(p).ravel()
    y = np.asarray(y).astype(bool).ravel()
    hard = p >= t
    tp = (hard & y).sum()
    fp = (hard & ~y).sum()
    fn = (~hard & y).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        'iou': float(tp / (tp + fp + fn + 1e-8)),
        'f1': float(2 * precision * recall / (precision + recall + 1e-8)),
        'precision': float(precision),
        'recall': float(recall),
        'auprc': float(average_precision_score(y, p)) if y.any() else float('nan'),
    }


_LOGF = None


def log(msg):
    """Print to the console and mirror to a per-seat log file.

    The console is what a human watches; the file is what survives the shell
    being closed, so both are written on every call.
    """
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if _LOGF is not None:
        _LOGF.write(line + '\n'); _LOGF.flush()


def open_log(path):
    """Start mirroring ``log`` output to ``path``."""
    global _LOGF
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOGF = open(path, 'a', encoding='utf-8')
    return _LOGF


# ---------------------------------------------------------------------------
# seats
# ---------------------------------------------------------------------------
def build_boehm(pretrained=True, encoder=ENCODER):
    """Boehm et al. 2022, iprapas/landslide-sar-unet @ e35fad9.

    Official recipe: smp.UnetPlusPlus(resnet50), CrossEntropyLoss, Adam.
    We keep the topology and the CE objective (method-specific), and follow
    protocol v2 for everything else. The official code relies on smp's
    default ``encoder_weights='imagenet'``; we make that explicit.

    The trunk is resnet34 rather than the official resnet50 so that every
    encoder-based seat sits at one capacity. Pass ``encoder='resnet50'`` to
    reproduce the published setting.
    """
    import segmentation_models_pytorch as smp
    net = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=5, classes=2,
    )
    return net, 'ce'


def build_cdnette(pretrained=True):
    """CDNetE Early Fusion (Bralet 2024, HAL tel-05029007) — no official code.

    A faithful early-fusion reading: target-orbit pre/post SAR channels are
    concatenated before a single-stream encoder, single two-class decoder.
    Same family as Boehm but a plain U-Net decoder, so the pair isolates
    decoder topology under one shared fusion contract.
    """
    import segmentation_models_pytorch as smp
    net = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet' if pretrained else None,
        in_channels=5, classes=2,
    )
    return net, 'ce'


def build_fc_siam():
    """FC-Siam-diff, rcdaudt/fully_convolutional_change_detection @ 4dd8323.

    The official ``SiamUnet_diff`` is kept verbatim; only the input split
    changes: 5 channels -> pre=[VV_pre,VH_pre,orbit], post=[VV_post,VH_post,orbit],
    with the orbit indicator repeated in both dates so the symmetric
    shared-weight topology is untouched. Output is log-probability, so the
    loss is NLL — that is the official objective, not a deviation.
    """
    sys.path.insert(0, str(THIRD / 'fc_siam_diff'))
    from siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(input_nbr=3, label_nbr=2), 'nll'


def build_mfewf(pretrained=True):
    """MFEWF (Chen et al. 2024, DOI 10.1080/17538947.2024.2393261) — no official code.

    Documented modules only: a residual backbone, AMM (channel+spatial
    attention on low level), CAASP (dilated multi-scale context on high
    level), MFFRM (adaptive low/high fusion). ResNet-34 trunk, ImageNet
    when available.
    """
    from external_mfewf import MFEWF
    return MFEWF(pretrained=pretrained), 'ce'


SEATS = {
    'boehm':   dict(build=build_boehm,   label='Boehm SAR U-Net++'),
    'cdnette': dict(build=build_cdnette, label='CDNetE Early Fusion'),
    'mfewf':   dict(build=build_mfewf,   label='MFEWF adapted'),
}

# FC-Siam is excluded from the v4 run by user decision (2026-09-19).  The builder
# below is kept so the earlier revision stays traceable, but the seat is no
# longer in SEATS and the CLI refuses it by name with this reason.
EXCLUDED_SEATS = {
    'fc_siam': "FC-Siam-diff is excluded from the v4 run by user decision "
               "('除了 fc siam 不跑'). No v4 number will exist for this seat.",
}


# ---------------------------------------------------------------------------
# forward / loss adapters — one place where a seat's interface differs
# ---------------------------------------------------------------------------
def forward_seat(net, seat, x):
    if seat == 'fc_siam':
        # x is (N,5,H,W): [VV_pre, VH_pre, VV_post, VH_post, orbit]
        pre = torch.cat((x[:, 0:1], x[:, 1:2], x[:, 4:5]), dim=1)
        post = torch.cat((x[:, 2:3], x[:, 3:4], x[:, 4:5]), dim=1)
        return net(pre, post)                       # (N,2,H,W) log-prob
    return net(x)


def loss_seat(logits, target, kind, posweight=None):
    """target is float (N,H,W) in {0,1}; logits is (N,2,H,W)."""
    y = target.long()
    if kind == 'ce':
        return F.cross_entropy(logits, y)
    if kind == 'nll':
        w = None
        if posweight is not None:
            w = torch.tensor([1.0, float(posweight)], device=logits.device, dtype=logits.dtype)
        return F.nll_loss(logits, y, weight=w)
    raise ValueError(kind)


def prob_seat(logits, kind):
    if kind == 'nll':
        return logits.exp()[:, 1]                   # already log-prob
    return F.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def monitor_prob(net, seat, kind, dataset):
    """Collect class-1 probabilities and labels over ``dataset``.

    Used both by the per-epoch monitor and by the final one-shot test read, so
    the number the early stop acts on and the number finally reported are
    produced by literally the same code path on the same data.
    """
    net.eval(); ps, ys = [], []
    for a, d, y, _ga, _gd, _real in loader(dataset, shuffle=False):
        x = torch.cat((a, d)).to(DEV)
        ps.append(prob_seat(forward_seat(net, seat, x), kind).cpu().numpy())
        ys.append(torch.cat((y, y)).numpy())
    return np.concatenate(ps), np.concatenate(ys).astype(int)


def monitor_auprc(net, seat, kind, dataset):
    """The monitored quantity: threshold-free AUPRC over the whole partition."""
    p, y = monitor_prob(net, seat, kind, dataset)
    return float(metric(p.ravel(), y.ravel(), t=THRESHOLD)['auprc'])


# ---------------------------------------------------------------------------
# one training run
# ---------------------------------------------------------------------------
def run_seat(seat, pretrained=True, encoder=None, resume=False):
    cfg = SEATS[seat]
    out = OUT_ROOT / seat
    out.mkdir(parents=True, exist_ok=True)

    seed()
    if seat == 'boehm':
        net, kind = cfg['build'](pretrained=pretrained, encoder=encoder or ENCODER)
    elif seat in ('mfewf',):
        net, kind = cfg['build'](pretrained=pretrained)
    else:
        net, kind = cfg['build']()
    net = net.to(DEV)
    n_param = sum(p.numel() for p in net.parameters()) / 1e6

    tid, vid, eid = split()
    if vid:
        raise SystemExit(f'{len(vid)} validation locations found; the frozen protocol has no '
                         'validation partition (expect COCD_SPLIT_DIR to hold only train/test)')
    train_set = HaitiPairs(tid, True)
    test_set = HaitiPairs(eid, False, train_set.stats)

    posweight = positive_weight(train_set)
    # The dataset indexes (location, mode) pairs — the orbit dimension is not in
    # ``spec`` because every __getitem__ returns BOTH orbits and the training
    # step concatenates them.  So one epoch forwards
    #     len(spec) * 2 = 1370 * 3 * 2 = 8220
    # single-orbit views, over ceil(len(spec)/batch) optimizer steps.
    n_views = len(train_set.spec) * 2
    if n_views != EXPECTED_TRAIN_VIEWS:
        raise SystemExit(f'{len(tid)} train locations give {n_views} views '
                         f'({len(train_set.spec)} pairs x 2 orbits), expected '
                         f'{EXPECTED_TRAIN_VIEWS} (1370 x 3 x 2)')
    log(f'{cfg["label"]}: {n_param:.4f} M, loss={kind}, seat={seat}, '
        f'train={len(train_set.spec)} pairs = {n_views} views '
        f'({len(tid)} locations x 3 modes x 2 orbits), test={len(test_set.spec)}')

    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    tl = loader(train_set, shuffle=True)

    # ---- early-stopping state -------------------------------------------------
    # best = the best monitored score seen so far ("第零个" in the user's wording)
    # bad  = how many consecutive epochs have failed to beat it
    # best_state = the weights at that best epoch; these, not the last epoch's,
    #              are the model the seat finally reports.
    best, bad, best_state, best_epoch = -1.0, 0, None, 0
    start_ep, hist = 1, []
    stop_reason = None
    latest = out / 'latest.pt'
    if resume and latest.exists():
        st = torch.load(latest, map_location=DEV, weights_only=False)
        net.load_state_dict(st['model']); opt.load_state_dict(st['optimizer'])
        start_ep = st['epoch'] + 1; hist = st.get('hist', [])
        best = st.get('best', -1.0); bad = st.get('bad', 0)
        best_epoch = st.get('best_epoch', 0); best_state = st.get('best_state')
        log(f'  resumed at epoch {start_ep} (best={best:.4f} @ep{best_epoch}, bad={bad})')

    for ep in range(start_ep, MAX_EPOCHS + 1):
        net.train(); losses = []
        t0 = time.time()
        bar = tqdm(tl, desc=f'{cfg["label"]} {ep:02d}/{MAX_EPOCHS}', unit='step',
                   leave=False, file=sys.stdout)
        for a, d, y, ga, gd, real in bar:
            x = torch.cat((a, d)).to(DEV)
            yy = torch.cat((y, y)).to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = loss_seat(forward_seat(net, seat, x), yy, kind, posweight)
            if not torch.isfinite(loss):
                raise SystemExit(f'non-finite loss at epoch {ep}')
            loss.backward(); opt.step(); losses.append(float(loss))
            bar.set_postfix(loss=f'{np.mean(losses):.4f}', refresh=False)
        bar.close()
        mean_loss = float(np.mean(losses))
        ep_secs = time.time() - t0

        # ---- the monitored score, then the stop test -------------------------
        # ONE score is read per epoch, on the test partition, threshold-free.
        # The threshold never enters this decision.
        score = monitor_auprc(net, seat, kind, test_set) if EARLY_STOP else float('nan')
        improved = (not EARLY_STOP) or (score > best)
        if improved:
            best, bad, best_epoch = score, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            flag = 'NEW BEST'
        else:
            bad += 1
            flag = f'no gain ({bad}/{PATIENCE})'

        hist.append(dict(epoch=ep, loss=mean_loss, monitor=score,
                         best=best, bad=bad, improved=bool(improved)))
        eta = ep_secs * (MAX_EPOCHS - ep)
        log(f'  epoch {ep:3d}/{MAX_EPOCHS}  loss={mean_loss:.5f}  '
            f'{MONITOR}={score:.4f}  best={best:.4f}@{best_epoch}  [{flag}]  '
            f'{ep_secs:.1f}s/epoch  ETA-max {eta/60:.1f}min')

        # latest.pt is the *resume* state, not the model; it tracks the last
        # epoch so an interrupted run can continue. The model is saved below.
        torch.save({'model': net.state_dict(), 'optimizer': opt.state_dict(),
                    'epoch': ep, 'hist': hist, 'best': best, 'bad': bad,
                    'best_epoch': best_epoch, 'best_state': best_state}, latest)
        (out / 'progress.json').write_text(json.dumps(
            {'seat': seat, 'epoch': ep, 'max_epochs': MAX_EPOCHS, 'loss': mean_loss,
             'monitor': MONITOR, 'monitor_value': score, 'best': best,
             'best_epoch': best_epoch, 'bad': bad, 'patience': PATIENCE,
             'improved': bool(improved), 'history': hist,
             'model': 'best_monitored_epoch', 'early_stop': bool(EARLY_STOP),
             'monitor_risk': MONITOR_RISK}, indent=2))

        if EARLY_STOP and bad >= PATIENCE:
            stop_reason = (f'{PATIENCE} consecutive epochs without beating '
                           f'{best:.4f} at epoch {best_epoch}')
            log(f'  EARLY STOP at epoch {ep}: {stop_reason}')
            break
    else:
        stop_reason = f'reached the {MAX_EPOCHS}-epoch ceiling'

    # the best-monitored-epoch weights ARE the model
    if best_state is None:                               # only if EARLY_STOP is off
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_epoch = MAX_EPOCHS
    net.load_state_dict(best_state)
    net.to(DEV)
    torch.save(net.state_dict(), out / 'best.pt')
    log(f'  model = epoch {best_epoch} weights, monitored {MONITOR}={best:.4f}; '
        f'stopped by: {stop_reason}')

    # one-shot test read, threshold fixed at 0.5
    p, yv = monitor_prob(net, seat, kind, test_set)
    gts, gcs, oids = [], [], []
    with torch.no_grad():
        for a, d, _y, ga, gd, _real in loader(test_set, shuffle=False):
            gts.append(torch.cat((ga, gd)).numpy()); gcs.append(torch.cat((gd, ga)).numpy())
            oids.append(np.concatenate([np.zeros(a.shape[0]), np.ones(d.shape[0])]).astype(int))
    gt = np.concatenate(gts); gc = np.concatenate(gcs); oid = np.concatenate(oids)
    np.savez(out / 'test_predictions.npz', p=p, y=yv, gt=gt, gc=gc, orbit=oid)

    rows = []
    for part, sel in [('overall', np.ones(len(p), bool)),
                      ('asc', oid == 0), ('desc', oid == 1)]:
        m = metric(p[sel].ravel(), yv[sel].ravel(), t=THRESHOLD)
        rows.append(dict(method=cfg['label'], partition=part, threshold=THRESHOLD,
                         selected_epoch=best_epoch, stopped_by=stop_reason, **m))
    import csv
    with open(out / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for r in rows:
        log(f'  TEST {r["partition"]:8s} IoU@0.5={r["iou"]:.4f} F1@0.5={r["f1"]:.4f} '
            f'AUPRC={r["auprc"]:.4f}')
    log(f'{cfg["label"]} done -> {out}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seat', default='all', help='boehm | cdnette | mfewf | all')
    ap.add_argument('--no-pretrained', action='store_true',
                    help='disable ImageNet weights (the old external-seat behaviour)')
    ap.add_argument('--encoder', default=None,
                    help=f'override the Boehm trunk (default {ENCODER}; use resnet50 for the '
                         'published setting)')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    # An excluded seat is refused by name, with its reason, rather than reported
    # as an unknown seat -- the distinction matters when reading a transcript.
    if args.seat in EXCLUDED_SEATS:
        raise SystemExit(f'{args.seat!r} is excluded from the v4 run: '
                         f'{EXCLUDED_SEATS[args.seat]}')
    seats = list(SEATS) if args.seat == 'all' else [args.seat]
    for s in seats:
        if s not in SEATS:
            raise SystemExit(f'unknown seat {s!r}; known seats: {sorted(SEATS)}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f'=== {PROT["revision"]} unified protocol: {len(seats)} seat(s) {seats} ===')
    log(f'Adam lr={LR:g} wd={WD} batch={BATCH} ceiling={MAX_EPOCHS} '
        f'early_stop={EARLY_STOP} patience={PATIENCE} monitor={MONITOR}, '
        f'threshold fixed {THRESHOLD}, encoder={args.encoder or ENCODER}, '
        f'pretrained={not args.no_pretrained}')
    log(f'protocol revision {PROT["revision"]}, frozen {PROT["frozen"]}')
    log(f'!! {MONITOR_RISK}')
    log(f'train {EXPECTED_TRAIN_VIEWS} views/epoch, test 686 views, output {OUT_ROOT}')
    if EXCLUDED_SEATS:
        log(f'excluded from this run: {", ".join(sorted(EXCLUDED_SEATS))}')

    done = []
    for i, s in enumerate(seats, 1):
        log(f'--- [{i}/{len(seats)}] {SEATS[s]["label"]} ---')
        open_log(LOG_DIR / f'{s}.log')
        try:
            rows = run_seat(s, pretrained=not args.no_pretrained,
                            encoder=args.encoder, resume=args.resume)
            done.append((s, rows))
        except SystemExit as e:
            log(f'!! {s} aborted: {e}')
            raise
        finally:
            if _LOGF is not None:
                _LOGF.close()
                globals()['_LOGF'] = None

    log(f'=== all {len(done)} seat(s) finished ===')
    for s, rows in done:
        r = next(x for x in rows if x['partition'] == 'overall')
        log(f'  {SEATS[s]["label"]:22s} IoU@0.5={r["iou"]:.4f} F1@0.5={r["f1"]:.4f} '
            f'AUPRC={r["auprc"]:.4f}')


if __name__ == '__main__':
    main()
