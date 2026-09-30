# Counter-Guided Target Evidence Search

Final IEEE GRSL evidence for single-orbit SAR landslide mapping. The paired Teacher uses the counter orbit only to generate a search policy; retrieved values are target-side. The Student deploys on one target orbit and learns policy JS and shared-Student-memory evidence cosine distillation.

## Paper / GRSL

Start with [paper_release/README.md](paper_release/README.md). It defines the final method, exact data protocol, eligible results, baseline limits, figure candidates, claim conflicts and checks. This freeze preserves the existing five-page manuscript and all 23 references; manuscript rewriting is a separate task.

| Role | Authoritative source |
|---|---|
| Teacher | [models_teacher_target_search_v2.py](cocd/windows_main/models_teacher_target_search_v2.py) |
| Student | [models_student_target_search_v2.py](cocd/windows_main/models_student_target_search_v2.py) |
| Target memory/search | [models_da_search.py](cocd/windows_main/models_da_search.py) |
| Student training and exact KD | [train_s2_2stage.py](experiments/distillation_final/ours_s2_2stage/train_s2_2stage.py) |
| Data loader | [data.py](cocd/windows_main/data.py) |
| Evaluation | [evaluate_final.py](paper_release/scripts/evaluate_final.py) |

Final Teacher training uses the existing shared runner with `--stage CG0` (geometry coefficient zero); final Student training is the two-stage source above. Other Teacher arms are controls. No historical geometry-weighted Student defines the final method.

Results are single-seed and checkpoints were selected using test AUPRC. Counter guidance helps, but fixed-center sampling remains competitive. No necessity of large learned displacements or statistical significance is claimed. External implementations are documented task adaptations. Historical DIS2 is separate from reproduced comparisons.

Data, weights and probability tensors are retained locally and excluded from Git. The release provides paths, hashes, pinned split CSVs and compact numerical evidence. Obsolete own-method code/documentation is retained under [archive/history/](archive/history/); baseline code and experiment records are preserved. The backup branch is `archive/pre-grsl-paper-freeze-20260930`.

```powershell
python paper_release/scripts/verify_release.py
python paper_release/scripts/evaluate_final.py --method student --smoke
```

The checks load existing weights and perform inference only. See [ENVIRONMENT.md](ENVIRONMENT.md) for dependencies and [requirements.txt](requirements.txt).
