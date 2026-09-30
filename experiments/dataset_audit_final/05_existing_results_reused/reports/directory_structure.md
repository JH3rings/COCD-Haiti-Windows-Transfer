# Directory Structure Audit — Haiti Multimodal Landslide Dataset

**Dataset:** *Multimodal Remote Sensing Dataset for Landslide Change Detection in Haiti*
Bralet A., Trouvé E., Chanussot J., Atto A.M., IEEE Dataport, 2024-07-14, doi: [10.21227/4heb-7h07](https://dx.doi.org/10.21227/4heb-7h07)
**Location on disk (already extracted; no `.tar.gz` present):** `/Users/zhangjiuqi/Desktop/distortion/`
**Audit working dir:** `/Users/zhangjiuqi/Desktop/distortion/Haiti_SAR_GRSL_Audit/`

> Note: the task expected `Pre_event.tar.gz / Post_event.tar.gz / Annotations.tar.gz` in `~/Downloads`.
> Downloads contained only an unrelated `sar-ship-dataset-master.zip`. The **already-extracted** dataset was
> found in the selected project folder `Desktop/distortion/`. No archives needed to be unpacked; the original
> files were only read, never modified. Raw-archive references therefore point at the extracted folders.

## 1. Top-level layout (source, read-only)

```
distortion/
├── Pre_event/                      # 3 acquisition folders, 1713 .tif each (5139 files, 1.5 GB)
│   ├── S1_ASC_20210805/{id}.tif    # Sentinel-1 ASCENDING  pre-event  (2021-08-05)
│   ├── S1_DESC_20210803/{id}.tif   # Sentinel-1 DESCENDING pre-event  (2021-08-03)
│   └── S2_20210804/{id}.tif        # Sentinel-2 optical    pre-event  (2021-08-04)
├── Post_event/                     # 3 acquisition folders, 1713 .tif each (5139 files, 1.5 GB)
│   ├── S1_ASC_20210817/{id}.tif    # Sentinel-1 ASCENDING  post-event (2021-08-17)
│   ├── S1_DESC_20210815/{id}.tif   # Sentinel-1 DESCENDING post-event (2021-08-15)
│   └── S2_20210814/{id}.tif        # Sentinel-2 optical    post-event (2021-08-14)
└── Annotations/{id}.tif            # 1713 NASA landslide label maps (114 MB), NOT georeferenced
```

## 2. Sizes / counts

| Item | Value |
|---|---|
| Samples (IDs) | **1713**, contiguous `0 … 1712`, identical ID set in every folder |
| SAR files (S1) | 6852 (4 views/dates × 1713) |
| Optical files (S2) | 3426 (2 dates × 1713) |
| Annotation files | 1713 |
| **Total files** | **11991 GeoTIFFs** (only extension present; no README/XML/JSON/sidecar shipped) |
| Logical size | 3,261,877,470 B ≈ **3.26 GB** (≈3.1 GB on disk) |
| Patch size | **128 × 128 px** for every file (uniform, already patchised, ~10 m/px, EPSG:4326) |

## 3. Naming rules

- Folder: `S1_<ORBIT>_<YYYYMMDD>` (`S1` = Sentinel-1 SAR; `ASC`/`DESC`), `S2_<YYYYMMDD>` = Sentinel-2.
- File: `<sample_id>.tif`, zero-padded **not** used (`0.tif … 1712.tif`).
- Sample ID is the join key: identical integer across all 7 locations → **strict 1:1 correspondence, no orphans**.

## 4. Per-file internal structure (the "what is in one sample" answer)

| File | bands | dtype | CRS | content |
|---|---|---|---|---|
| `S1_ASC_* / S1_DESC_* .tif` | **3** | float32 | EPSG:4326 | **band1 = VH (dB)**, **band2 = VV (dB)**, **band3 = layover/shadow geometry mask {0,1,2,3}** |
| `S2_*.tif` | 4 | float64 | EPSG:4326 | band1-3 = RGB reflectance, band4 = cloud/cloud-shadow binary mask |
| `Annotations/{id}.tif` | 1 | float32 | **none (identity)** | NASA landslide label {0,1,2,3}; `>0` = landslide |

Full machine-readable inventory: **`metadata/file_inventory.csv`** (columns: sample_id, event_period, sensor,
nominal_direction, relative_path, filename, extension, size_bytes, count, width, height, dtypes, crs,
transform, resx, resy, bounds, nodata).

## 5. Audit working directory (all derived artefacts)

```
Haiti_SAR_GRSL_Audit/
├── raw_archives/        # empty (no tar.gz found; sources are the extracted folders, left untouched)
├── extracted/           # empty (data already extracted in place)
├── scripts/             # 00…10 reproducible python scripts
├── metadata/            # file_inventory.csv, acquisition_metadata.csv, temporal_pairs.csv, global_categorical_summary.json
├── qc/tables/           # independence, leakage, complementarity, registration, baseline, completeness, repeatability…
├── qc/figures/          # figureA…F + registration/ (12 checkerboard/edge panels)
├── processed/           # 1713 sample_XXXXXX/ clean training-ready index (VV + geom + landslide + metadata.json), 629 MB
├── manifests/           # dataset_manifest.csv, split_manifest.csv (70/15/15 by sample, seed 42)
└── reports/             # directory_structure.md, final_haiti_dataset_audit.md
```
Added footprint ≈ **653 MB** (well under the 2–3 GB target; original 3.1 GB never copied/modified).
