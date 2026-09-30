# Dataset and protocol provenance

- Dataset root: `data/haiti`; locked split: `data/splits_v3`; 1,370 merged training locations and 343 test locations. No validation split was used.
- Each location has ASC and DESC Sentinel-1 pre/post paired inputs; the loader reads bands `[2, 1]` as `[VV, VH]` for each date and appends an orbit indicator, yielding `[pre VV, pre VH, post VV, post VH, orbit_id]`. Test evaluation expands each location into ASC and DESC orbit-conditioned pixels (686 patches, 11,239,424 pixels).
- Geometry masks are the existing `asc_pre_geom.tif` and `desc_pre_geom.tif`. Internal `G10` is retained as `TGD = target mask positive AND counter mask negative`; `Remaining` is its valid-domain complement.
- Batch size 16, Adam, learning rate 5e-5, maximum 100 epochs, patience 2, threshold 0.5. Test AUPRC was used for development monitoring/checkpoint selection; these are not untouched held-out estimates.
- Metric aggregation is pooled over ASC and DESC orbit-conditioned evaluation pixels. AUPRC is sklearn average precision; IoU/F1/precision/recall are pooled positive-class metrics at threshold 0.5.
- Acquisition dates, CRS, georeferencing and reliable feature-to-mask coordinate inversion were not exposed by the locked analysis contract; they are listed as missing rather than inferred.
