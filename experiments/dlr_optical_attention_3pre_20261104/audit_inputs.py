"""Check the released DLR tensors without changing official split membership."""
import csv
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "dlr_landslide_audit" / "audit"
SAR_TIMES = ("PRE1", "PRE2", "PRE3", "POST1")
# Select visible, broad NIR and both SWIR bands from the released S2 fields.
OPTICAL_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


def read_rows(name):
    with (AUDIT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    scenes = {
        (row["EID"], row["beginposition"][:10]): row
        for row in read_rows("scene_acquisition_metadata.csv")
    }
    samples = read_rows("sample_manifest.csv")
    counts = {}
    results = []
    for filename in sorted({row["h5_path"] for row in samples}):
        with h5py.File(filename, "r") as archive:
            sar_fields = [f"{time}_{band}" for time in SAR_TIMES for band in ("gVV", "gVH")]
            optical_fields = [f"{time}_{band}" for time in ("PRE1", "POST1") for band in OPTICAL_BANDS]
            for field in [*sar_fields, *optical_fields, "None_MASK"]:
                assert archive[field].shape == archive[sar_fields[0]].shape, field
            for sample in (row for row in samples if row["h5_path"] == filename):
                idx = int(sample["h5_index"])
                stored_id = np.asarray(archive.attrs["IDs_order"][idx]).item()
                if isinstance(stored_id, bytes):
                    stored_id = stored_id.decode()
                assert stored_id == sample["sample_id"], (stored_id, sample["sample_id"])
                sar = np.stack([archive[key][idx, 0] for key in sar_fields])
                masks = (np.isfinite(sar) & (sar > 0)).reshape(4, 2, *sar.shape[-2:]).all(axis=1)
                optical = np.stack([archive[key][idx, 0] for key in optical_fields])
                optical_valid = np.isfinite(optical).all(axis=0)
                dates = [datetime.strptime(sample[field + "_date"], "%d.%m.%Y").date() for field in sar_fields]
                event = date.fromisoformat(sample["date"])
                chronology = dates[7] > event and all(day < event for day in dates[:6])
                paired = dates[::2] == dates[1::2] and len(set(dates[:6:2])) == 3
                matched = [scenes.get((sample["event_id"], day.isoformat())) for day in dates[::2]]
                verified_track = all(matched) and len({
                    (row["relativeorbitnumber"], row["orbitdirection"]) for row in matched
                }) == 1
                first_confirmed = sample["satellite_actual_first_post_confirmed"].lower() == "true"
                joint = masks.all(axis=0)
                split_counts = counts.setdefault(sample["official_split"], Counter())
                flags = {
                    "samples": True,
                    "calendar_3plus1": chronology and paired and sample["orbit"] in ("ASC", "DES"),
                    "same_calendar_day_post": dates[6] == event,
                    "relative_orbit_verified": verified_track,
                    "actual_first_post_verified": first_confirmed,
                    "nonempty_joint_sar_valid": joint.any(),
                    "exact_sar_footprint_match": sample["footprint_status"] == "EXACT_FULL_SAR_PIXEL_MATCH",
                }
                split_counts.update({key: int(value) for key, value in flags.items()})
                results.append({
                    "sample_id": sample["sample_id"], "split": sample["official_split"],
                    "event": sample["event_id"], "direction": sample["orbit"],
                    "event_date": str(event), **dict(zip(SAR_TIMES, map(str, dates[::2]))),
                    **{key: bool(value) for key, value in flags.items()},
                    "valid_3plus1_fraction": float(joint.mean()),
                    "valid_1plus1_fraction": float((masks[0] & masks[3]).mean()),
                    "optical_finite_fraction": float(optical_valid.mean()),
                })
    assert len(results) == 563 and len({row["sample_id"] for row in results}) == 563
    with (HERE / "sample_input_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "by_split": counts, "optical_bands": OPTICAL_BANDS,
        "mean_joint_valid_3plus1": float(np.mean([r["valid_3plus1_fraction"] for r in results])),
        "mean_joint_valid_1plus1": float(np.mean([r["valid_1plus1_fraction"] for r in results])),
        "official_split_changed": False,
    }
    (HERE / "input_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
