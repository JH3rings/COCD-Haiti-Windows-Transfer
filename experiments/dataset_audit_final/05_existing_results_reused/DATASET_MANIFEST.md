# Dataset manifest

## 1. Where the data is

| | |
|---|---|
| original location | `/Users/zhangjiuqi/Desktop/distortion/` — the four `Pre_event/`·`Post_event/` acquisition folders, plus `Haiti_SAR_GRSL_Audit/processed/` |
| in this package | `<package>/data/haiti/` |
| expected on Windows | unpacking the zip reproduces `data/haiti/` verbatim; nothing has to be moved |
| override | set `COCD_DATASET_ROOT` to point somewhere else (e.g. a larger drive) — `cocd/paths.py` reads it |

The dataset itself is **included in this zip** (1,372.6 MB across 13,704 files).
The originals on the Mac are untouched.

```
data/haiti/
├── Pre_event/
│   ├── S1_ASC_20210805/    1713 files, 322.0 MB    ASC pre-event,  2021-08-05
│   └── S1_DESC_20210803/   1713 files, 322.0 MB    DESC pre-event, 2021-08-03
├── Post_event/
│   ├── S1_ASC_20210817/    1713 files, 322.0 MB    ASC post-event,  2021-08-17
│   └── S1_DESC_20210815/   1713 files, 322.0 MB    DESC post-event, 2021-08-15
├── processed/
│   └── sample_000000 … sample_001712/
│       ├── landslide_mask.tif    128×128 uint8, 0 background, 1/2/3 inventory class
│       ├── asc_pre_geom.tif      128×128 uint8, 0 valid, 1 layover, 2 shadow, 3 both
│       ├── desc_pre_geom.tif     same coding, counter orbit
│       └── metadata.json         per-location provenance (see §6)
└── DATA_MANIFEST.csv       13704 rows: relative_path, size_bytes, sha256
```

## 2. The unit of the study is a **spatial location**

1,713 locations, `sample_id` 0 … 1712. Each location owns four acquisitions and
two masks. **Everything belonging to one location stays in one partition** — the
split is made on `sample_id`, and the loader reads all four acquisitions of a
location at once, so a location physically cannot straddle two splits.

## 3. The split — pinned, never re-drawn

| partition | locations | file |
|---|---:|---|
| train | **1,233** | `data/splits/train_ids.csv` |
| validation (internal) | **137** | `data/splits/val_ids.csv` |
| test | **343** | `data/splits/test_ids.csv` |

- The three files contain the explicit `sample_id` lists, **in the exact order the
  existing models were trained on**. `cocd/scripts/21_train_rapid_landslide_cocd.py::split()`
  reads them in preference to any random draw, so the partition cannot change on
  the target machine. Delete or rename them and the loader falls back to the
  original seed-42 draw — do not do that.
- Provenance: `cocd/manifests/spatial_80_20_split.csv` holds the upstream 80/20
  call (**1,370 train / 343 test**, seed 42), and the 137 validation locations are
  a seed-42 draw of 10 % of the 1,370 (`numpy.random.default_rng(42).choice(...,
  size=137, replace=False)`). 1,233 + 137 = 1,370.
- SHA256 of the pinned lists:

  | file | sha256 |
  |---|---|
  | `train_ids.csv` | `297f5e2e4aee8753137efa08084b6dedf1cc1ed981e2ac0e95fcdac6f03b1d2b` |
  | `val_ids.csv` | `08ebf5e4e71f51bcff828cb5fb5897d668e6ce653b1f7b18de9de55df4f88d8d` |
  | `test_ids.csv` | `31528c417f361a85b7ba7fd7ecf8a2fdb62f70cff07400a0fce33aa9ea66fb2b` |

## 4. What the loader expects, exactly

`scripts/21_train_rapid_landslide_cocd.py`, `HaitiPairs.__init__`:

```python
RAW = {('asc','pre') : 'Pre_event/S1_ASC_20210805',
       ('desc','pre'): 'Pre_event/S1_DESC_20210803',
       ('asc','post'): 'Post_event/S1_ASC_20210817',
       ('desc','post'):'Post_event/S1_DESC_20210815'}

for o in ('asc', 'desc'):
    for e in ('pre', 'post'):
        z[(o, e)] = read(DATASET_ROOT / RAW[(o, e)] / f'{sid}.tif', [2, 1])

land = read(DATASET_ROOT / f'processed/sample_{sid:06d}' / 'landslide_mask.tif') > 0
ga   = read(DATASET_ROOT / f'processed/sample_{sid:06d}' / 'asc_pre_geom.tif')
gd   = read(DATASET_ROOT / f'processed/sample_{sid:06d}' / 'desc_pre_geom.tif')
```

Naming rules that follow from this:

- raw file is `{sample_id}.tif` with **no zero padding** (`809.tif`, not `000809.tif`);
- processed directory is `sample_{sample_id:06d}` — **zero-padded to six digits**;
- the two polarisations are **not** separate files: the raw stack has three bands,
  **band 1 = VH, band 2 = VV**, band 3 unused, and `read(..., [2, 1])` returns them
  as `[VV, VH]`;
- the geometry mask of the **post** acquisition is not read at all; the regions are
  defined from the two **pre** masks (target-orbit mask and counter-orbit mask);
- one label per location, used for both orbits.

The five-channel input is assembled in `HaitiPairs.__getitem__`:

```
[VV_pre, VH_pre, VV_post, VH_post, orbit]     128 × 128, float32
orbit = a spatially constant plane, 0.0 for ASC and 1.0 for DESC
```

and standardised with the mean/std of the **train** partition only, which is
computed once from the train locations and passed to validation and test.

## 5. Raw file format

| property | value |
|---|---|
| size | 197,094 bytes each |
| raster | 128 × 128, 3 bands, float32, uncompressed, little-endian GeoTIFF |
| CRS | EPSG:4326 |
| resolution | 10 m (`8.983152841e-05` degrees) |
| band 1 | VH |
| band 2 | VV |
| band 3 | unused by this study |

## 6. `metadata.json` — read the caveat

Each `processed/sample_XXXXXX/metadata.json` records the split, the four
acquisition dates, the geometry class coding, and the source paths of the files
it was built from. **Those source paths are absolute macOS paths**
(`/Users/zhangjiuqi/Desktop/distortion/...`). They are provenance only: **no code
in this package reads that file**, and the training pipeline never opens it. It is
kept because it documents which upstream file each location came from. Do not
"fix" it by rewriting the paths — it would stop being a record of what was built.

## 7. What is deliberately not in this package

| not shipped | why |
|---|---|
| `Pre_event/S2_20210804/`, `Post_event/S2_20210814/` (Sentinel-2 optical, 1.7 GB) | optical is an input the student is explicitly forbidden to use; it belongs to a different experiment line |
| `Annotations/` (171.3k files, 114 MB) | the upstream NASA inventory rasters; `processed/landslide_mask.tif` is derived from them and is what the loader reads |
| `processed/sample_*/{asc,desc}_{pre,post}_vv.tif` (456 MB) | single-band VV extracts used only by the geometry-prediction experiment line (`scripts/11`, `scripts/13`), not by the COCD learner — the learner reads the two polarisations from the raw stacks instead |
| `processed/sample_*/{asc,desc}_post_geom.tif` (66 MB) | the regions are defined from the pre-event masks only |
| `qc/`, `raw_archives/`, `extracted/` | dataset QC figures and empty staging folders |

Excluding those does not change any number in the COCD line: the four files per
location that this package ships are exactly the ones `HaitiPairs` opens.

## 8. Verifying the data on Windows

**Counts and sizes** — from the package root:

```powershell
python -c "import pandas as pd;d=pd.read_csv('data/haiti/DATA_MANIFEST.csv');print(len(d), d.size_bytes.sum()/1048576)"
```
expected: `13704 1372.62…`

**Per-file integrity** (recommended after any copy over a network or a USB disk):

```powershell
python - <<'PY'
import csv, hashlib, pathlib
root = pathlib.Path('data/haiti')
bad = []
for rel, size, want in csv.reader(open(root / 'DATA_MANIFEST.csv')):
    if rel == 'relative_path':
        continue
    p = root / rel
    if not p.exists():
        bad.append((rel, 'missing')); continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != want:
        bad.append((rel, 'sha256 mismatch'))
print('checked', sum(1 for _ in open(root / 'DATA_MANIFEST.csv')) - 1, 'files;', len(bad), 'problems')
for b in bad[:20]:
    print(' ', b)
PY
```

**Structural check that the loader can actually consume it**
`python verify_transfer.py`, section [4] and [5]. It reads one location end to
end, builds a batch, and pushes it through the encoder and the decoder.

**Expected per-directory totals**

| directory | files | MB |
|---|---:|---:|
| `Pre_event/S1_ASC_20210805` | 1,713 | 322.0 |
| `Pre_event/S1_DESC_20210803` | 1,713 | 322.0 |
| `Post_event/S1_ASC_20210817` | 1,713 | 322.0 |
| `Post_event/S1_DESC_20210815` | 1,713 | 322.0 |
| `processed/` (3 masks/geometries + 1 json per location) | 6,852 | 84.7 |
| **total** | **13,704** | **1,372.6** |

## 9. QC facts worth knowing before training

- **Landslide prevalence.** The mask values are `{0, 1, 2, 3}` — the NASA
  inventory class, with `> 0` meaning landslide. Per location the positive count
  varies widely: 260 px (1.59 %) at location 0, 151 px at 100, 877 px at 500,
  4,724 px at 809, 165 px at 1500. Across the whole train partition the positive
  fraction of all label pixels is about 6.5 %. This is why the first epochs are
  the window in which a background collapse would be visible: the protocol uses
  plain BCE with no class reweighting, so a run whose validation AUPRC sits at
  the prevalence has collapsed.
- **Masks are spatially distinct, not a repeated template.** Locations 0–3 share
  the same class counts (16124/191/69) because they are neighbouring grid cells
  of one inventory raster, but their per-file SHA256 differ and their transform
  origins differ (`-74.243101` vs `-74.237351`).
- **Geometry distortion is unevenly distributed.** Location 0 has 767 distorted
  pixels out of 16,384 in the ASC pre mask; the four partitions `G00`/`G01`/
  `G10`/`G11` are populated very unevenly at dataset scale (`G00` 9,058,636 px,
  `G10` = `G01` = 1,039,406 px, `G11` 101,976 px), which is why a per-region
  breakdown is reported alongside the aggregate.
