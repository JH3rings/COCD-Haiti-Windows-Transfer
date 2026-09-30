# Counter-Guided Target Evidence Search

Single-orbit SAR landslide mapping with paired-orbit training. A Counter-Guided Teacher uses complementary observations to choose and aggregate target-side evidence. A target-only Student transfers this behavior through Policy Distillation and Shared-Memory Evidence Distillation, and deploys on one orbit.

- **Teacher:** target + counter determine the policy; target features supply searched values.
- **Student:** target only, with no counter image, Teacher or geometry mask at inference.
- **Policy transfer:** spatial sampling distributions, Jensen-Shannon divergence.
- **Evidence transfer:** both policies search the same Student target memory, cosine alignment.

[Architecture](docs/final_architecture.md) and [reproduction settings](docs/method_description.md) describe the final method. This branch contains our method code and lightweight documentation. Data, model weights, external comparison adapters, experiment results and manuscript assets remain local.

```powershell
pip install -r requirements.txt
python cocd/windows_main/smoke.py
python cocd/windows_main/evaluate.py --checkpoint <student.pt> --output <metrics.json>
```

Training sources are `cocd/windows_main/train_teacher.py` and `train_student.py`. Default local inputs are `data/haiti`, `data/splits_v3`, and `artifacts/ours`; dataset/device overrides are supported by `cocd/paths.py`. The input contract is pre/post VV/VH plus orbit indicator. Geometry is used only for regional evaluation.
