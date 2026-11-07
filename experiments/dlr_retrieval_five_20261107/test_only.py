"""Evaluate the completed seed42 queue after the user's replication stop."""
import json
import traceback

from run import OUT, OLD, REFERENCE, METHODS, common, selection
from final_evaluation import finish


def main():
    common.setup()
    for method in METHODS:
        selection(OUT/f"{method}_student_seed42")
    for method in ("M0","M1","M3","M4"):
        for role in ("sar","teacher"):
            selection(OUT/f"{method}_{role}_seed42")
    common.write(OUT/"validation_method_selection.json",{
        "eligible":list(METHODS),"seed43_repeats":[],
        "reason":"User requested immediate test of all five completed seed42 methods."})
    frames=common.frames()
    stats=json.loads((REFERENCE/"normalization.json").read_text())
    optical=json.loads((OLD/"optical_normalization.json").read_text())
    finish(list(METHODS),[],frames,stats,optical)
    common.write(OUT/"status.json",{"state":"COMPLETE","scope":"seed42 only; extra seeds stopped by user"})


if __name__=="__main__":
    try:
        main()
    except Exception:
        common.write(OUT/"status.json",{"state":"FAILED","error":traceback.format_exc()})
        raise
